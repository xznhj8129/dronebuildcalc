import math
import warnings
import pandas as pd
import aerosandbox as asb
import aerosandbox.numpy as np
import neuralfoil as nf
import json
import argparse
from simlib import *

# Silence NumPy scalar-conversion deprecation spam
warnings.filterwarnings("ignore", category=DeprecationWarning)












def create_elliptical_fuselage(
    length: float,
    diameter: float,
    cap_len: float,
    n_cap: int = 12,
    n_cyl: int = 1,
) -> asb.Fuselage:
    R      = diameter / 2
    xsecs  = []

    def push(tag: str, x: float, r: float):
        if xsecs and abs(x - xsecs[-1].xyz_c[0]) < 1e-9:
            return                                      # no duplicates, no steps
        print(f"{tag:6s}  x:{x:8.3f}  r:{r:6.3f}")
        xsecs.append(asb.FuselageXSec(xyz_c=[x, 0, 0], radius=r))

    # ── nose cap (tip → tube-join, *exclude* join itself) ─────────────────────
    for t in np.linspace(0.0, 1.0, n_cap, endpoint=False):
        x = t * cap_len
        r = R * math.sqrt(1 - (1 - t) ** 2)             # quarter-ellipse
        push("NOSE", x, r)

    # ── straight tube interior (no ends) ──────────────────────────────────────
    span = length - (2 * cap_len)
    print(span)
    for i in range(1, 2):
        x = cap_len + span * i / (2)
        push("TUBE", x, R)

    # ── tail cap (tube-join → tail-tip, *include* tip) ────────────────────────
    base = length - cap_len
    push("TUBE", base, R)                               # single join point
    for t in np.linspace(0.0, 1.0, n_cap + 1)[1:]:
        x = base + t * cap_len
        r = R * math.sqrt(1 - t ** 2)                   # mirrored quarter-ellipse
        push("TAIL", x, r)

    return asb.Fuselage(name="Fuselage", xsecs=xsecs)





def create_surface(
    idx: int,
    span: float,
    chord_root: float,
    chord_tip: float,
    x_le: float,
    y_le: float,
    z_le: float,
    ang: float,
    aoa_deg: float,
    sweep_deg: float,
    name_prefix: str,
    airfoil_name: str,
) -> asb.Wing:
    base_af = asb.Airfoil(airfoil_name)
    dx_sweep = span * math.tan(math.radians(sweep_deg))

    theta = math.radians(ang)
    y_tip = span * math.cos(theta)
    z_tip = span * math.sin(theta)

    root = asb.WingXSec(
        xyz_le=[x_le, y_le, z_le],
        chord=chord_root,
        twist=0, ###
        airfoil=base_af,
        control_surface_is_symmetric=False,
        control_surface_deflection=0.0,
    )
    tip = asb.WingXSec(
        xyz_le=[x_le+dx_sweep, y_tip, z_tip],
        chord=chord_tip,
        twist=0,
        airfoil=base_af,
    )
    return asb.Wing(
        name=f"{name_prefix}_{idx+1}",
        symmetric=True,
        xsecs=[root, tip],
    )

# ==================================================
# BASIC FLIGHT & AIRFRAME PARAMETERS
# ==================================================

with open("configs.json","r") as file:
    assembly_configs = json.loads(file.read())

with open("modules.json","r") as file:
    modules = json.loads(file.read())

with open("parts.json","r") as file:
    parts = json.loads(file.read())


batt_v = {
  "lipo": {
    "max_V": 4.20,
    "nominal_V": 3.70,
    "min_V": 3.00
  },
  "liion": {
    "max_V": 4.20,
    "nominal_V": 3.60,
    "min_V": 2.50
  }
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parametric airframe sim')
    parser.add_argument('--assembly', default='assembly_plane', help="Assembly configuration to load")
    parser.add_argument('--altitude', type=float, default=100, help="Sim altitude")
    parser.add_argument('--airspeed', type=float, default=30, help="Sim airspeed")
    parser.add_argument('--aoa', type=float, default=0.0, help="Angle of attack")
    parser.add_argument('--view', action='store_true', help="Show 3-view")
    parser.add_argument('--vlm', action='store_true', help="Show vortex lattice simulation")
    parser.add_argument('--export_vsp', action='store_true', help="VSP file export")
    arguments = parser.parse_args()

    # Couple Assembly
    chasm = assembly_configs[arguments.assembly]
    assembly = []
    for m in chasm["parts"]:
        assembly.append(m)
        coupling = {"module":"joint-ring-c"}
        assembly.append(coupling)
    assembly.pop()

    canards = chasm["canards"]

    base_amps = 1.0

    # ==================================================
    # SIMULATION PARAMETERS
    # ==================================================
    altitude_m       = arguments.altitude    # [m]
    airspeed         = arguments.airspeed     # [m/s]
    aoa_deg          = arguments.aoa

    # ==================================================
    # ASSEMBLY & MASS/LENGTH/CG SUMMARY
    # ==================================================
    wing_sections        = []
    fuselage_length    = 0.0
    total_mass_kg        = 0.0
    total_moment_kg_m    = 0.0  # Sum of (mass_i * cg_x_i)
    current_x_fuselage_m = 0.0  # Tracks the x-coordinate of the FRONT of the current module
    tube_diam = 0.0
    body_x_start = 0.0
    body_x_end = 0.0

    print("="*120)
    print(f"Airframe: {arguments.assembly}")
    print("="*120)
    header = (f"{'Module/Part Name':<25} | {'Type':<10} | {'Mass (kg)':>9} | {'Len (m)':>7} | "
            f"{'Local CG (m)':>12} | {'Global CG (m)':>13} | {'Mod Front_x (m)':>14}")
    print(header)
    print("-" * len(header))
    num_motors = 0

    for i, item_in_assembly in enumerate(assembly):
        module_name = item_in_assembly["module"]
        module_data = modules[module_name]
        if tube_diam ==0:
            tube_diam = module_data["diam"]
        module_mass = module_data["mass"]
        module_len  = module_data["len"]
        module_cg_spec_val = module_data["cg"] # Value from 'cg' field in module definition

        module_cg_offset_from_module_front_m = module_cg_spec_val * module_len

        # Absolute CG of the module's structural mass (in global fuselage reference frame)
        module_structure_cg_global_x_m = current_x_fuselage_m + module_cg_offset_from_module_front_m

        # Print module's own contribution details
        print(f"{module_name:<25} | {module_data['type']:<10} | {module_mass:>9.4f} | {module_len:>7.4f} | "
            f"{module_cg_offset_from_module_front_m:>12.4f} | {module_structure_cg_global_x_m:>13.4f} | {current_x_fuselage_m:>14.4f}")

        # Accumulate module's own mass and moment contribution
        total_mass_kg     += module_mass
        total_moment_kg_m += module_mass * module_structure_cg_global_x_m

        # --- Process parts/attachments associated with this module ---
        # Default CG location for most attached parts is the module's structural CG
        attached_part_cg_global_x_m = module_structure_cg_global_x_m

        if module_data["type"] == "wingmount":
            wing_spec_name = item_in_assembly["surfaces"]
            wing_data = modules[wing_spec_name]
            num_wings = module_data["wings_n"]
            wing_mass_each = wing_data["mass"]
            total_wings_mass = num_wings * wing_mass_each
            wing_sections.append(i)

            print(f"{'  ∟ Wing: '+wing_spec_name:<25} | {'surface':<10} | {total_wings_mass:>9.4f} | {'N/A':>7} | "
                f"{'(at mount CG)':>12} | {attached_part_cg_global_x_m:>13.4f} |")

            total_mass_kg     += total_wings_mass
            total_moment_kg_m += total_wings_mass * attached_part_cg_global_x_m

            if wing_data.get("motor", False):
                motor_name = item_in_assembly["motors"]
                prop = item_in_assembly["props"]
                brand, model, kv = motor_name.split("_")
                motor_model = MotorData(brand, model, prop, kv, 6)
                num_motors_per_wing = wing_data["motor_n"]
                num_motors = num_wings * num_motors_per_wing
                motor_mass_each = motor_model.motor_weight  / 1000.0
                total_wing_motors_mass = num_motors * motor_mass_each
                
                print(f"{'    ∟ Motor: '+motor_name:<25} | {'on wing':<10} | {total_wing_motors_mass:>9.4f} | {'N/A':>7}"
                    f"{'(at mount CG)':>12} | {attached_part_cg_global_x_m:>13.4f}")
                print(f"{'    ∟ Propeller: '+prop:<25} ")

                total_mass_kg     += total_wing_motors_mass
                total_moment_kg_m += total_wing_motors_mass * attached_part_cg_global_x_m


        elif module_data["type"] == "body" and module_data.get("payload", False):
            payload_mass = assembly[i]["payload_mass"]

            print(f"{'  ∟ Payload: ':<25} | {'in body':<10} | {payload_mass:>9.4f} | {'N/A':>7} | "
                f"{'(at body CG)':>12} | {attached_part_cg_global_x_m:>13.4f} |")

            total_mass_kg     += payload_mass
            total_moment_kg_m += payload_mass * attached_part_cg_global_x_m

        elif module_data["type"] == "body" and module_data.get("battery", False):
            battery_name = item_in_assembly["battery"]
            battery_part_data = parts["battery"][battery_name]
            batt_S = battery_part_data['s']
            batt_P = battery_part_data['p']
            if battery_part_data['type'] == "lipo":
                batt_mah = battery_part_data['mah']
                battery_mass = battery_part_data["mass"]
                batt_curr = (battery_part_data['mah'] / 1000) * battery_part_data['c']
                batt_nominal_v = 3.7
            elif battery_part_data['type'] == "liion":
                celltype = battery_part_data['cell']
                batt_S = battery_part_data['s']
                batt_P = battery_part_data['p']
                batt_data = LiIonBatteryData(*celltype, batt_S, batt_P)
                battery_mass = batt_data.pack_weight
                batt_mah = batt_data.pack_amps * 1000
                batt_curr = batt_data.pack_current
                batt_nominal_v = batt_data.nominal_v
            

            print(f"{'  ∟ Battery: '+battery_name:<25} | {'in body':<10} | {battery_mass:>9.4f} | {'N/A':>7} | "
                f"{'(at body CG)':>12} | {attached_part_cg_global_x_m:>13.4f} |")
            print(f"  ∟ Capacity: {batt_mah:.0f} mah | Current limit: {batt_curr:.0f} A")

            total_mass_kg     += battery_mass
            total_moment_kg_m += battery_mass * attached_part_cg_global_x_m

        elif module_data["type"] == "tail" and module_data.get("motor", False):
            motor_name = item_in_assembly["motors"]
            prop = item_in_assembly["props"]
            brand, model, kv = motor_name.split("_")
            motor_model = MotorData(brand, model, prop, int(kv), int(6))
            num_motors = module_data["motor_n"]
            motor_mass_each = motor_model.motor_weight / 1000.0
            total_tail_motors_mass = num_motors * motor_mass_each

            # Tail motors have specific placement via motor_x_pos (ratio of TAIL module length)
            motor_x_pos_ratio_in_tail = module_data.get("motor_x_pos", 0.5) # Default to mid-module if not specified
            motor_cg_offset_from_module_front_m = module_len * motor_x_pos_ratio_in_tail
            tail_motor_cg_global_x_m = current_x_fuselage_m + motor_cg_offset_from_module_front_m
            
            print(f"{'  ∟ Motor: '+motor_name:<25} | {'in tail':<10} | {total_tail_motors_mass:>9.4f} | {'N/A':>7} | "
                f"{motor_cg_offset_from_module_front_m:>12.4f} | {tail_motor_cg_global_x_m:>13.4f} |")
            print(f"{'    ∟ Propeller: '+prop:<25} |")

            total_mass_kg     += total_tail_motors_mass
            total_moment_kg_m += total_tail_motors_mass * tail_motor_cg_global_x_m

        # Update fuselage total length and the x-coordinate for the front of the NEXT module
        fuselage_length    += module_len
        current_x_fuselage_m += module_len
        if i < len(assembly) -1 : # Avoid printing separator after last item
            print("-" * len(header))

    # motor data:
    static_thrust = (motor_model.max_thrust/1000.0) * num_motors
    static_thrust_N = static_thrust * 9.81

    # Calculate final CG
    cg_x_m = 0.0
    if total_mass_kg > 0:
        cg_x_m = total_moment_kg_m / total_mass_kg
    else:
        print("\nWarning: Total mass is zero, CG cannot be calculated.")

    weight_N  = total_mass_kg * 9.81
    print()
# ==================================================
    # WING & TAIL MODULES SUMMARY
    # ==================================================
    #Winged section 1 servo-canards-1: 4 x canard-1
    #print(wing_sections)
    #print()
    for i, sec in enumerate(wing_sections):
        #print(i,sec)
        st = assembly[sec]["module"]
        sd = assembly[sec]
        #print('st',st)
        mn = modules[st]
        #print('mn',mn)
        sw = sd["surfaces"]
        #print('sw',sw)
        w = modules[sw]
        print(f"Winged section {i+1} {st}: {mn['wings_n']} x {sw}")
        print(f"\tairfoil     = {w['airfoil']}")
        print(f"\tspan        = {w['span']:.3f} m")
        print(f"\tchord root  = {w['chord_root']:.3f} m")
        print(f"\tchord tip   = {w['chord_tip']:.3f} m")
        print(f"\tsweep       = {w['sweep_ang']:.1f}°")
        print(f"\tmount ang   = {mn['wings_ang']:.1f}°")
        print()


    # ==================================================
    # PRECOMPUTE MODULE POSITIONS
    # ==================================================
    x = 0.0
    module_x_map = []
    for mn in assembly:
        mod = mn["module"]
        length = modules[mod]["len"]
        module_x_map.append(x + length / 2)
        x += length
    # ==================================================
    # BUILD FUSELAGE FROM MODULES
    # ==================================================
    n_cap = 12            # pts per hemispherical cap
    EPS   = 1e-6          # duplicate-merge tolerance

    # ---- PASS 1 : locate ends of the straight tube -----------------------------
    body_x_start = body_x_end = 0.0
    x_cursor = 0.0
    for mn in assembly:
        m = modules[mn["module"]]
        if m["type"] == "nose":
            body_x_start = x_cursor + m["len"]          # tube starts after nose
        elif m["type"] == "tail":
            body_x_end   = x_cursor                     # tube ends where tail starts
        x_cursor += m["len"]

    body_length = body_x_end - body_x_start
    print(f"body_x_start: {body_x_start:.3f} m")
    print(f"body_x_end:   {body_x_end:.3f} m")
    print(f"body_length:  {body_length:.3f} m")

    # ---- PASS 2 : generate X-sections ------------------------------------------
    xsecs    = []
    x_cursor = 0.0

    def add_xsec(x, r):
        """Append an X-section only if it is at a new x-station."""
        if xsecs and abs(x - xsecs[-1].xyz_c[0]) < EPS:
            return
        xsecs.append(asb.FuselageXSec(xyz_c=[x, 0, 0], radius=r))

    for mn in assembly:
        m  = modules[mn["module"]]
        L  = m["len"]
        R  = m["diam"] / 2

        if m["type"] == "nose":
            add_xsec(x_cursor, 0.0)                                   # tip
            for xi in np.linspace(0, L, n_cap + 2)[1:-1]:             # skip duplicates
                r = R * math.sqrt(1 - ((xi - L) / L) ** 2)
                add_xsec(x_cursor + xi, r)
            add_xsec(x_cursor + L, R)                                 # joint to tube

        elif m["type"] in ("body", "wingmount"):
            add_xsec(x_cursor, R)
            add_xsec(x_cursor + L, R)

        elif m["type"] == "tail":
            add_xsec(x_cursor, R)                                     # tube → cap
            for xi in np.linspace(0, L, n_cap + 2)[1:-1]:
                r = R * math.sqrt(1 - (xi / L) ** 2)
                add_xsec(x_cursor + xi, r)
            add_xsec(x_cursor + L, 0.0)                               # tail tip

        x_cursor += L

    fuselage = asb.Fuselage(name="Fuselage", xsecs=xsecs)

    # ==================================================
    # BUILD WINGS & TAIL-FINS FROM MOUNTS
    # ==================================================
    main_wings = []
    tail_fins  = []
    wing_chord_root = 0
    wing_chord_tip = 0
    tail_chord_root = 0
    tail_chord_tip = 0
    wing_type = ""
    wing_section = ""
    tail_type = ""
    tail_section = ""
    wing_n = 0
    tail_n = 0
    wing_span = 0
    tail_span = 0

    n_wing_modules = 0
    #wing_sections
    print("X pos map:", module_x_map)
    for i, mn in enumerate(assembly):
        m = modules[mn["module"]]
        #print(mn)
        fuselage_radius = m["diam"] / 2
        if m["type"] == "wingmount":
            spec_name = mn["surfaces"]
            wdata     = modules[spec_name]
            panel_count = m["wings_n"] // 2
            n_wing_modules += 1

            for j in range(panel_count):
                aoa = wdata["aoa_ang"]
                flip = 1 if j==0 else -1
                print(n_wing_modules, wdata)
                wingang = m["wings_ang"] * flip
                
                x_pos = module_x_map[i] - (wdata["chord_root"]/2)
                y_pos = (m["diam"] / 2) * math.cos(math.radians(wingang))
                z_pos = (m["diam"] / 2) * math.sin(math.radians(wingang))
                wing = create_surface(
                    idx=j,
                    span=wdata["span"] + fuselage_radius, # TODO: Possible incongruity: span is calculated from center, which shows smaller span when fuselage taken into account
                    chord_root=wdata["chord_root"],
                    chord_tip=wdata["chord_tip"],
                    x_le=x_pos,
                    y_le=y_pos,
                    z_le=z_pos,
                    ang=wingang,
                    aoa_deg=aoa,
                    sweep_deg=wdata["sweep_ang"],
                    name_prefix=spec_name,
                    airfoil_name=wdata["airfoil"],
                )

                if (n_wing_modules == 1 and not canards) or (n_wing_modules == 2 and canards):
                    wing_type = spec_name
                    wing_section = mn
                    wing_chord_root = wdata["chord_root"]
                    wing_chord_tip = wdata["chord_tip"]
                    wing_n = m["wings_n"]
                    wing_span = wdata["span"]
                    wing_airfoil = wdata["airfoil"]
                    wing_aoa = wdata["aoa_ang"]
                    main_wings.append(wing)

                elif (n_wing_modules == 1 and canards) or (n_wing_modules == 2 and not canards):
                    tail_type = spec_name
                    tail_section = mn
                    tail_chord_root = wdata["chord_root"]
                    tail_chord_tip = wdata["chord_tip"]
                    tail_n = m["wings_n"]
                    tail_span = wdata["span"]
                    tail_airfoil = wdata["airfoil"]
                    tail_aoa = wdata["aoa_ang"]
                    tail_fins.append(wing)

    S_main_planform = float(wing_n * wing_span * (wing_chord_root + wing_chord_tip) / 2)
    S_tail_planform = float(tail_n * tail_span * (tail_chord_root + tail_chord_tip) / 2)


    # ==================================================
    # ASSEMBLE AIRPLANE
    # ==================================================
    airplane = asb.Airplane(
        name="Parametrized Airplane",
        xyz_ref=[0, 0, 0],  # Moment reference at nose
        s_ref=sum(w.area() for w in main_wings),
        c_ref=(
            wing_chord_root
            + wing_chord_tip
        ) / 2,
        b_ref=sum(w.span() for w in main_wings),
        wings=main_wings + tail_fins,
        fuselages=[fuselage],
    )

    if arguments.export_vsp:
        airplane.export_OpenVSP_vspscript(f"{arguments.assembly}.vspscript")
        #airplane.export_AVL(f'{arguments.export}.avl')
    # ==================================================
    # VORTEX-LATTICE ANALYSIS
    # ==================================================
    vlm = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(velocity=airspeed, alpha=aoa_deg),
        spanwise_resolution=25, chordwise_resolution=8
    )
    aero = vlm.run()

    print()
    print("VLM Analysis Raw Output")
    for k, airspeed in aero.items():
        print(f"{k.rjust(4)} : {airspeed}")
    print()

    # forces & coefficients from VLM
    L      = float(aero["L"])
    D_ind  = float(abs(aero["D"])) # Induced drag from VLM
    My     = float(aero["M_b"][1]) # Pitching moment about airplane.xyz_ref
    x_cp   = -My / L if L != 0 else float('nan') # Center of pressure relative to xyz_ref

    CL     = float(aero["CL"])
    CDi    = float(aero["CD"]) # Induced drag coefficient from VLM
    Cm     = float(aero["Cm"])

    # ==================================================
    # STABILITY DERIVATIVES
    # ==================================================
    stab_derivs = vlm.run_with_stability_derivatives()
    CLalpha_rad = stab_derivs["CLa"] # dCL/dalpha [rad^-1]
    Cmalpha_rad = stab_derivs["Cma"] # dCm/dalpha [rad^-1] about xyz_ref

    CLalpha_deg = CLalpha_rad * (math.pi / 180)
    Cmalpha_deg = Cmalpha_rad * (math.pi / 180)

    x_np = float('nan')
    if CLalpha_rad != 0:
        # airplane.xyz_ref[0] is the x-coordinate of the moment reference point (0 for nose)
        x_np = airplane.xyz_ref[0] - (Cmalpha_rad / CLalpha_rad) * airplane.c_ref

    static_margin_pct = float('nan')
    if not math.isnan(x_np) and airplane.c_ref != 0:
        static_margin_pct = ((x_np - cg_x_m) / airplane.c_ref) * 100

    # ==================================================
    # ATMOSPHERE & REYNOLDS
    # ==================================================
    atm  = asb.Atmosphere(altitude=altitude_m)
    rho  = atm.density()
    mu   = atm.dynamic_viscosity()
    airspeed    = vlm.op_point.velocity
    # Reynolds number based on main wing root chord for airfoil analysis
    Re_wing_root = rho * airspeed * wing_chord_root / mu
    Re_tail_root = rho * airspeed * tail_chord_root / mu

    # ==================================================
    # ZERO-LIFT PROFILE DRAG USING NeuralFoil
    # ==================================================
    CD0_main_airfoil = float(
        nf.get_aero_from_airfoil(
            airfoil=asb.Airfoil(wing_airfoil), alpha=0, Re=int(Re_wing_root)
        )["CD"]
    )
    CD0_tail_airfoil = float(
        nf.get_aero_from_airfoil(
            airfoil=asb.Airfoil(tail_airfoil), alpha=0, Re=int(Re_tail_root)
        )["CD"]
    )

    # planform & wetted areas
    S_ref  = airplane.s_ref
    S_fuse_wet = fuselage.area_wetted() # More accurate wetted area

    # Wetted area factors (approx. 2 for thin airfoils)
    # For more accuracy: factor = 2 * (1 + t_c_mean + 1.2*(t_c_mean)^2 + ... )
    # Problem: Assumes t/c = 0.09 (NACA6409) for main wings and 0.08 (NACA0008) for tail fins. Matches the airfoils in the outputs, but hardcoding limits flexibility if airfoils change. AeroSandbox’s Airfoil class doesn’t provide thickness directly, so this is a practical simplification.
    # Assuming NACA6409 t/c = 0.09, NACA0008 t/c = 0.08
    # TODO: Except we might not be using those precise airfoils, FIX THIS


    main_wing_wetted_area_factor = 2.0 * (1 + 0.09 + 1.2 * (0.09**2)) # More precise approx.
    tail_fin_wetted_area_factor = 2.0 * (1 + 0.08 + 1.2 * (0.08**2))   # More precise approx.

    S_main_wetted = S_main_planform * main_wing_wetted_area_factor
    S_tail_wetted = S_tail_planform * tail_fin_wetted_area_factor

    # flat-plate fuselage drag
    Re_fuse = rho * airspeed * fuselage_length / mu
    # Cf_fuse_turbulent = 0.074 / (Re_fuse ** 0.2) # Turbulent flat plate (Schlichting)
    # Cf_fuse_laminar = 1.328 / (math.sqrt(Re_fuse)) # Laminar flat plate
    # For mixed flow, a common one: 0.455 / (math.log10(Re_fuse) ** 2.58) (ITTC line)
    Cf_fuse = 0.455 / (math.log10(Re_fuse) ** 2.58) if Re_fuse > 1e5 else 1.328 / (math.sqrt(Re_fuse))

    form_factor_fuse = 1 + 60 / ((fuselage_length / tube_diam)**3) + 0.0025 * (fuselage_length / tube_diam) # Hoerner

    CD0_contrib_fuse = Cf_fuse * form_factor_fuse * S_fuse_wet / S_ref

    # Component contributions to CD0
    CD0_contrib_main = CD0_main_airfoil * S_main_wetted / S_ref
    CD0_contrib_tail = CD0_tail_airfoil * S_tail_wetted / S_ref

    CD0_components = {
        "MainWings": CD0_contrib_main,
        "TailFins" : CD0_contrib_tail,
        "Fuselage" : CD0_contrib_fuse
    }
    C_D0 = sum(CD0_components.values())

    # Optional: Add interference drag (e.g., 5-15% of component sum)
    C_D0_interf_factor = 0.10 # ???????
    # Problem: This is a crude estimate. Interference drag varies with geometry (e.g., wing-fuselage junctions, motor placement). A flat 10% may over- or underestimate drag, especially for the quadrocket with wingtip motors.
    C_D0 *= (1 + C_D0_interf_factor)

    # TODO: propeller drag

    # ==================================================
    # MAX CL FROM NeuralFoil SWEEP (for main wing airfoil)
    # ==================================================
    alphas_clmax_sweep = np.linspace(-5, 20, 51) # NeuralFoil default range is good
    CLs_airfoil = np.array([
        nf.get_aero_from_airfoil(
            airfoil=asb.Airfoil(wing_airfoil),
            alpha=float(a),
            Re=int(Re_wing_root),
        )["CL"]
        for a in alphas_clmax_sweep
    ])
    CL_max_airfoil_2D = float(CLs_airfoil.max())
    alpha_stall_airfoil_2D = float(alphas_clmax_sweep[CLs_airfoil.argmax()])

    # Estimate 3D CLmax (rough approximation, typically 0.8-0.9 of 2D for finite wings)
    CL_max_3D_factor = 0.85 # TODO: based on what?
    CL_max_eff_3D = CL_max_airfoil_2D * CL_max_3D_factor

    # ==================================================
    # PERFORMANCE CALCULATIONS
    # ==================================================
    # Effective Aspect Ratio of the aircraft
    AR_eff = (airplane.b_ref**2 / airplane.s_ref) if airplane.s_ref != 0 else float('inf')

    e_oswald = 0.75  # AI: "Assumed Oswald efficiency factor for X-wing (can range 0.7-0.9)" (TODO: What if it isn't? NO ASSUMPTIONS)
    k_induced_drag = 1 / (math.pi * AR_eff * e_oswald) if AR_eff > 0 and e_oswald > 0 else float('inf')

    # Total drag coefficient at VLM operating point
    C_D_total_op_point = C_D0 + CDi # CDi is from VLM at the specific CL (aero["CL"])
    D_total_op_point   = float(C_D_total_op_point * 0.5 * rho * airspeed ** 2 * S_ref)

    # Stall speed
    V_stall   = float(math.sqrt(2 * weight_N / (rho * S_ref * CL_max_eff_3D))) if CL_max_eff_3D > 0 and rho > 0 and S_ref > 0 else float('inf')

    #### [CHATGPTO3 FOLLOW]

    # ── 1. Propeller efficiency curve (crude bell-shape) ─────────────────────────
    def prop_eta(V, V_pitch):
        """
        Returns propulsive efficiency 0.15 → 0.8 → 0 as speed rises to pitch speed.
        """
        if V <= 0:                               # safety
            return 0.15
        V_pk   = 0.30 * V_pitch                  # η peaks here
        V_zero = V_pitch                         # thrust ~ 0 here
        if V < V_pk:                             # rising limb
            return 0.15 + 0.65 * V / V_pk
        if V < V_zero:                           # falling limb
            return 0.80 * (1 - (V - V_pk) / (V_zero - V_pk))
        return 0.0                               # beyond pitch speed

    # Estimate pitch speed from prop data

    prop_diam     = motor_model.prop_diameter * 0.0254    
    prop_pitch_m  = motor_model.prop_pitch * 0.0254    
    rpm_no_load   = motor_model.kv * batt_S * batt_nominal_v
    V_pitch   = rpm_no_load * prop_pitch_m / 60.0

    # ── 2. Drag & power required at speed V ──────────────────────────────────────
    def drag_power(V):
        """
        Returns:
            D  : drag / thrust required  [N]
            P  : shaft power required    [W]
            ηp : propulsive efficiency   [-]
        """
        q   = 0.5 * rho * V**2
        CL  = weight_N / (q * S_ref)
        CD  = C_D0 + k_induced_drag * CL**2
        D   = q * S_ref * CD
        ηp  = prop_eta(V, V_pitch)
        P   = D * V / max(ηp, 1e-3)             # avoid divide-by-zero
        return D, P, ηp

    # ─────────────────────────────────────────────────────────────────────────────
    #  OPTIMUM ENDURANCE & RANGE
    # ─────────────────────────────────────────────────────────────────────────────

    # 1) Speed sweep --------------------------------------------------------------
    V_min = max(1.05 * V_stall, 4.0)
    V_max = 10 * V_stall
    V_vec = np.linspace(V_min, V_max, 301)

    D_vec = np.zeros_like(V_vec)     # drag / thrust required  [N]
    P_vec = np.zeros_like(V_vec)     # shaft power required    [W]

    for i, V in enumerate(V_vec):
        D_i, P_i, _ = drag_power(V)
        D_vec[i] = D_i
        P_vec[i] = P_i

    Wh_per_m = P_vec / 3600.0 / V_vec   # energy per metre  [Wh/m]

    # 2) Locate minima ------------------------------------------------------------
    idx_end = int(np.argmin(P_vec))      # min power  → endurance
    idx_rng = int(np.argmin(Wh_per_m))   # min Wh/m   → range

    # 3) ENDURANCE point ----------------------------------------------------------
    V_end      = float(V_vec[idx_end])
    D_end_N    = float(D_vec[idx_end])
    P_air_end  = float(P_vec[idx_end])
    T_total_end_g = D_end_N * 1000.0 / 9.81          # aircraft total thrust [g-f]
    T_per_motor_end_g = T_total_end_g / num_motors   # per motor [g-f]

    P_m_end, I_m_end, gW_end, thr_end = motor_model.motor_perf_thrust(T_per_motor_end_g)
    I_tot_end = I_m_end * num_motors
    batt_t_end = batt_mah / (1000*(base_amps + I_tot_end)) * 60
    flt_range_end = (V_end * (batt_t_end*60)) / 1000.0

    # 4) RANGE point --------------------------------------------------------------
    V_rng      = float(V_vec[idx_rng])
    D_rng_N    = float(D_vec[idx_rng])
    P_air_rng  = float(P_vec[idx_rng])

    T_total_rng_g = D_rng_N * 1000.0 / 9.81
    T_per_motor_rng_g = T_total_rng_g / num_motors

    P_m_rng, I_m_rng, gW_rng, thr_rng = motor_model.motor_perf_thrust(T_per_motor_rng_g)
    I_tot_rng = I_m_rng * num_motors
    batt_t_rng = batt_mah / (1000*(base_amps + I_tot_rng)) * 60
    flt_range_rng = (V_rng * (batt_t_rng*60)) / 1000.0

    # sweep speed until Drag > Thrust_avail
    V_grid = np.linspace( 2.0, V_pitch*2, 800 )         # 2 m/s → just below V_pitch
    T_av, D_req = [], []
    for V in V_grid:
        # thrust AVAILABLE at speed V (all motors combined)
        if V >= V_pitch:
            thrust_available = 0.0
        else:
            frac = 1.0 - V / V_pitch
            thrust_available = static_thrust_N * frac * frac   # [N]

        # drag required (?) at speed V
        q  = 0.5 * rho * V**2
        CL = weight_N / (q * S_ref)
        CD = C_D0 + k_induced_drag * CL**2
        drag_required = q * S_ref * CD

        T_av.append( thrust_available )
        D_req.append( drag_required )

    T_av = np.array(T_av)
    D_req = np.array(D_req)

    # last speed where thrust ≥ drag  →  V_max
    mask   = T_av >= D_req
    if np.any(mask):
        V_max_level_flight = float( V_grid[mask][-1] )
    else:
        V_max_level_flight = float('nan')   # can’t even hover in level flight!


    #### [CHATGPTO3 ENDS]

    # Max speed in level flight (Thrust = Drag, Lift = Weight)
    # Solves A_vmax * q^2 - B_vmax * q + C_vmax = 0 for q = 0.5 * rho * airspeed^2
    # Assuming Thrust = static_thrust_N (constant with speed for this simple calc)
    A_vmax_eq = S_ref * C_D0
    B_vmax_eq = -static_thrust_N
    C_vmax_eq = (k_induced_drag * (weight_N**2) / S_ref) if S_ref > 0 else float('inf')

    V_max_level_flight = float('nan')
    if A_vmax_eq != 0: # Standard quadratic case
        discriminant_vmax = B_vmax_eq**2 - 4 * A_vmax_eq * C_vmax_eq
        if discriminant_vmax >= 0:
            # q must be positive. The physically relevant solution for Vmax is typically the larger positive root for q.
            q_Vmax1 = (-B_vmax_eq + math.sqrt(discriminant_vmax)) / (2 * A_vmax_eq)
            q_Vmax2 = (-B_vmax_eq - math.sqrt(discriminant_vmax)) / (2 * A_vmax_eq)
            
            q_Vmax_solutions = []
            if not isinstance(q_Vmax1, complex) and q_Vmax1 > 0: q_Vmax_solutions.append(q_Vmax1)
            if not isinstance(q_Vmax2, complex) and q_Vmax2 > 0: q_Vmax_solutions.append(q_Vmax2)
                
            if q_Vmax_solutions:
                q_Vmax = max(q_Vmax_solutions) # Max speed corresponds to the larger positive dynamic pressure
                if rho > 0:
                    V_max_level_flight = math.sqrt(2 * q_Vmax / rho)

    elif B_vmax_eq != 0 and C_vmax_eq != float('inf'): # Linear case (CD0 = 0, unlikely but handled)
        # -Thrust_static * q + k * W^2 / S_ref = 0
        # Thrust_static * q = k * W^2 / S_ref
        # q = (k * W^2 / S_ref) / Thrust_static
        q_Vmax = C_vmax_eq / (-B_vmax_eq) # C_vmax_eq is (k_induced_drag * (weight_N**2) / S_ref)
        if q_Vmax > 0 and rho > 0:
            V_max_level_flight = math.sqrt(2 * q_Vmax / rho)

    # Max speed if CL=0 (only CD0 acts - theoretical upper bound if lift wasn't needed)
    V_max_at_CL0 = float(math.sqrt(2 * static_thrust_N / (rho * S_ref * C_D0))) if C_D0 > 0 and rho > 0 and S_ref > 0 else float('inf')

    wing_loading      = float(weight_N / S_main_planform) if S_main_planform > 0 else float('nan')
    wing_to_tail_ratio = float(S_main_planform / S_tail_planform) if S_tail_planform > 0 else float('nan')

    # L/D at VLM operating point
    LD_op_point = L / D_total_op_point if D_total_op_point != 0 else float('inf')

    # Estimated Max L/D and CL for Max L/D (parabolic polar approx.)
    LD_max_approx = float('nan')
    CL_at_LD_max_approx = float('nan')
    if C_D0 > 0 and k_induced_drag > 0 and k_induced_drag != float('inf'):
        LD_max_approx = math.sqrt(1 / (4 * C_D0 * k_induced_drag))
        CL_at_LD_max_approx = math.sqrt(C_D0 / k_induced_drag)

    # Load Factor at VLM operating point
    load_factor_op_point = L / weight_N if weight_N > 0 else float('nan')

    # Glide Performance (based on parabolic drag polar approximation)
    best_glide_angle_rad = float('nan')
    best_glide_ratio     = float('nan')
    sink_rate_at_best_glide = float('nan')
    V_at_best_glide = float('nan')

    if not math.isnan(LD_max_approx) and LD_max_approx > 0:
        best_glide_angle_rad = math.atan(1 / LD_max_approx)
        best_glide_ratio = LD_max_approx
        if not (math.isnan(CL_at_LD_max_approx) or CL_at_LD_max_approx <= 0 or rho <= 0 or S_ref <= 0 or weight_N <= 0):
            V_at_best_glide = math.sqrt((2 * weight_N) / (rho * S_ref * CL_at_LD_max_approx))
            sink_rate_at_best_glide = V_at_best_glide * math.sin(best_glide_angle_rad) # or V_at_best_glide / LD_max_approx

    # Minimum Sink Rate (based on parabolic drag polar approximation)
    CL_min_sink_approx = float('nan')
    CD_min_sink_approx = float('nan')
    V_at_min_sink_approx = float('nan')
    min_sink_rate_approx = float('nan')
    LD_at_min_sink_approx = float('nan')

    if C_D0 > 0 and k_induced_drag > 0 and k_induced_drag != float('inf'):
        CL_min_sink_approx = math.sqrt(3 * C_D0 / k_induced_drag)
        CD_min_sink_approx = C_D0 + k_induced_drag * (CL_min_sink_approx**2)
        if CL_min_sink_approx > 0: # Ensure CL is positive
            LD_at_min_sink_approx = CL_min_sink_approx / CD_min_sink_approx
            if not (rho <= 0 or S_ref <= 0 or weight_N <= 0):
                V_at_min_sink_approx = math.sqrt((2 * weight_N) / (rho * S_ref * CL_min_sink_approx))
                min_sink_rate_approx = V_at_min_sink_approx * (CD_min_sink_approx / CL_min_sink_approx) # V_sink = V * D/L = V * CD/CL

    # Rate of Climb/Sink at VLM Operating Point (assuming static_thrust_N applies at V)
    # This is a strong assumption; thrust typically varies with speed.
    # --- Rate-of-climb / flight-path angle at the operating point ---
    ROC_op_point            = float('nan')
    climb_angle_op_point_rad = float('nan')

    if weight_N > 0:
        excess_thrust = static_thrust_N - D_total_op_point      #  T − D
        sin_gamma     = excess_thrust / weight_N                #  = sin(γ)  for small-angle theory

        # ── keep sin_gamma in a legal range ───────────────────────────
        if sin_gamma >= 1:          # more thrust than weight → vertical climb
            climb_angle_op_point_rad = math.pi / 2              # 90°
        elif sin_gamma <= -1:       # negative and large → vertical dive
            climb_angle_op_point_rad = -math.pi / 2             # –90°
        else:                       # normal case
            climb_angle_op_point_rad = math.asin(sin_gamma)

        # ROC = V · sin(γ) works for all three branches above
        ROC_op_point = airspeed * sin_gamma


    # ==================================================
    # RESULTS
    # ==================================================
    print("\n" + "="*40)
    print("OVERALL SUMMARY:")
    print("="*40)
    print(f"Wing AoA         : {wing_aoa:.3f} deg")
    print(f"Tail AoA         : {tail_aoa:.3f} deg")
    print(f"Total Fuselage Length : {fuselage_length:.4f} m")
    print(f"Total Mass            : {total_mass_kg:.4f} kg")
    print(f"Total N Mass          : {weight_N:.4f} N")
    print(f"Total Moment (kg*m)   : {total_moment_kg_m:.4f} kg*m")
    print(f"Center of Gravity (CG_x): {cg_x_m:.4f} m (from fuselage front, x=0)")
    print()
    print(f"Airspeed: {airspeed:.3f} m/s")
    print(f"Altitude: {altitude_m:.2f} m")
    print(f"Atm density: {rho:.3f} kg/m³")
    print(f"Reynolds (Main Wing Root): {Re_wing_root:.2e}")
    print(f"Reynolds (Tail Fin Root): {Re_tail_root:.2e}")
    print()
    print(f"--- Motors ---")
    print(f"Motor name             : {brand} {model} {kv}KV")
    print(f"Motor num:             : {num_motors}")
    print(f"Propellers:            : {prop}")
    print(f"Total max thrust       : {static_thrust*1000:.2f} g")
    print()

    print(f"--- VLM Operating Point Results (alpha_aircraft = {aoa_deg:.1f} deg, airspeed = {airspeed:.1f} m/s) ---")
    print(f"Lift (L)                                 = {L:.2f} N")
    print(f"Induced drag (Dᵢ)                        = {D_ind:.2f} N")
    print(f"Total drag at Op Point (Dₜ_op)           = {D_total_op_point:.2f} N")
    print(f"Pitching moment (Mᵧ about xyz_ref)       = {My:.2f} N·m")
    print()

    print(f"--- Areas ---")
    print(f"Reference Area (S_ref)                   = {S_ref:.4f} m² (Total main wing planform)")
    print(f"Main Wing Planform Area (S_main)         = {S_main_planform:.4f} m²")
    print(f"Tail Fin Planform Area (S_tail)          = {S_tail_planform:.4f} m²")
    print(f"Fuselage Wetted Area (S_fuse_wet)        = {S_fuse_wet:.4f} m²")
    print(f"Main Wing Wetted Area (S_main_wet)       = {S_main_wetted:.4f} m²")
    print(f"Tail Fin Wetted Area (S_tail_wet)        = {S_tail_wetted:.4f} m²")
    print()

    print(f"--- Coefficients at VLM Operating Point ---")
    print(f"Lift coefficient (C_L)                   = {CL:.4f}")
    print(f"Induced-drag coefficient (C_Dᵢ)          = {CDi:.4f}")
    print(f"Total drag coefficient (C_Dₜ_op)         = {C_D_total_op_point:.4f}")
    print(f"Pitching-moment coefficient (C_m)        = {Cm:.4f}")
    print()

    print(f"--- Zero-Lift Drag (C_D0) Breakdown ---")
    print(f"Main wing airfoil ({wing_airfoil}) CD @ Re={Re_wing_root:.1e}, α=0°: {CD0_main_airfoil:.5f}")
    print(f"Tail fin airfoil ({tail_airfoil}) CD @ Re={Re_tail_root:.1e}, α=0°:  {CD0_tail_airfoil:.5f}")
    print(f"Fuselage skin friction coeff (Cf_fuse)   = {Cf_fuse:.5f} (Re_fuse={Re_fuse:.1e})")
    print(f"Fuselage form factor (FF_fuse)           = {form_factor_fuse:.3f}")
    print(f"CD0 Contribution - Main Wings            = {CD0_components['MainWings']:.5f} ({(CD0_components['MainWings']/C_D0*100 if C_D0!=0 else 0):.1f}%)")
    print(f"CD0 Contribution - Tail Fins             = {CD0_components['TailFins']:.5f} ({(CD0_components['TailFins']/C_D0*100 if C_D0!=0 else 0):.1f}%)")
    print(f"CD0 Contribution - Fuselage              = {CD0_components['Fuselage']:.5f} ({(CD0_components['Fuselage']/C_D0*100 if C_D0!=0 else 0):.1f}%)")
    print(f"Interference Drag Factor Applied         = {C_D0_interf_factor*100:.1f}%")
    print(f"Total Zero-lift drag coeff (C_D₀)        = {C_D0:.5f}")
    print()

    print(f"--- Performance Estimates ---")
    print(f"Effective Aspect Ratio (AR_eff)          = {AR_eff:.2f}")
    print(f"Assumed Oswald Efficiency (e)            = {e_oswald:.2f}")
    print(f"Induced Drag Factor (k = 1/(π*AR*e))     = {k_induced_drag:.4f}")
    print(f"L/D at VLM Op Point                      = {LD_op_point:.2f}")
    print(f"Est. Max L/D (parabolic approx.)         = {LD_max_approx:.2f} (at CL ≈ {CL_at_LD_max_approx:.3f})")
    print(f"  Best Glide Ratio (L/D_max)             = {best_glide_ratio:.2f}")
    print(f"  Best Glide Angle                       = {math.degrees(best_glide_angle_rad) if not math.isnan(best_glide_angle_rad) else 'NaN':.2f} deg")
    print(f"  Airspeed for Best Glide (V_bg)         = {V_at_best_glide:.2f} m/s")
    print(f"  Sink Rate at Best Glide                = {sink_rate_at_best_glide:.2f} m/s")
    print(f"Est. Min Sink Rate (parabolic approx.)   = {min_sink_rate_approx:.2f} m/s")
    print(f"  Airspeed for Min Sink (V_ms)         = {V_at_min_sink_approx:.2f} m/s (at CL ≈ {CL_min_sink_approx:.3f})")
    print()

    print("--- Max endurance speed ---")
    print(f"V {V_end:.2f} m/s")
    print(f"Thurst req {T_per_motor_end_g:.2f} g")
    print(f"P_air {P_air_end:.2f} W")
    print(f"P/motor {P_m_end:.2f} W")
    print(f"I/motor {I_m_end:.2f} A")
    print(f"I_total {I_tot_end:.2f} A")
    print(f"g/W {gW_end:.2f}")
    print(f"Thr {thr_end:.0f} %")
    print(f"Flight time {batt_t_end:.2f} min ({batt_t_end/60.0:.2f} h)")
    print(f"Flight range: {flt_range_end:.2f} km")
    print()

    print("--- Max range speed ---")
    print(f"V {V_rng:.2f} m/s")
    print(f"Thurst req {T_per_motor_rng_g:.2f} g")
    print(f"P_air {P_air_rng:.2f} W")
    print(f"P/motor {P_m_rng:.2f} W")
    print(f"I/motor {I_m_rng:.2f} A")
    print(f"I_total {I_tot_rng:.2f} A")
    print(f"g/W {gW_rng:.2f}")
    print(f"Thr {thr_rng:.0f} %")
    print(f"Flight time {batt_t_rng:.2f} min ({batt_t_rng/60.0:.2f} h)")
    print(f"Flight range: {flt_range_rng:.2f} km")
    print()

    #print("-"*106)

    print(f"Max level-flight speed (with thrust fall-off) : {V_max_level_flight: .1f} m/s {V_max_level_flight*3.6:.1f} km/h")

    print(f"\n--- Performance at VLM Operating Point (V={airspeed:.1f}m/s, T_static={static_thrust_N:.1f}N) ---")
    print(f"Load Factor (L/W) at Op Point          = {load_factor_op_point:.2f}")
    print(f"Rate of Climb/Sink at Op Point         = {ROC_op_point:.2f} m/s")
    print(f"Climb/Descent Angle at Op Point        = {math.degrees(climb_angle_op_point_rad) if not math.isnan(climb_angle_op_point_rad) else 0:.2f} deg\n")
    print()

    print(f"Main wing airfoil 2D C_Lmax ({wing_airfoil}) = {CL_max_airfoil_2D:.4f} at α = {alpha_stall_airfoil_2D:.1f}° (Re={Re_wing_root:.1e})")
    print(f"Estimated 3D C_Lmax (aircraft)           = {CL_max_eff_3D:.4f} (using factor {CL_max_3D_factor:.2f} on 2D C_Lmax)")
    print(f"Stall speed (Vₛ at W={weight_N:.1f}N)            = {V_stall:.2f} m/s")
    print(f"Max speed (V_max, level flight)          = {V_max_level_flight:.2f} m/s (assuming T_static={static_thrust_N:.1f}N)")
    print(f"Max speed (V_max @ CL=0, theoretical)    = {V_max_at_CL0:.2f} m/s")
    print()

    print(f"--- Stability & Control (at VLM Op Point) ---")
    print(f"CG Location (x_cg from nose)             = {cg_x_m:.3f} m")
    print(f"Center of pressure (x_cp from xyz_ref)   = {x_cp:.3f} m")
    print(f"Neutral Point (x_np from nose)           = {x_np:.3f} m")
    print(f"Static Margin (SM)                       = {static_margin_pct:.1f} %c_ref")
    print(f"Lift curve slope (CLα)                   = {CLalpha_deg:.4f} /deg")
    print(f"Moment curve slope (Cmα about xyz_ref)   = {Cmalpha_deg:.4f} /deg")
    print()

    print(f"--- General Parameters ---")
    print(f"Wing loading (W/S_main)                  = {wing_loading:.2f} N/m²")
    print(f"Wing-to-tail area ratio (S_main/S_tail)  = {wing_to_tail_ratio:.2f}")
    print()

    # ==================================================
    # VISUALISATION
    # ==================================================
    
    if arguments.view: 
        airplane.draw_three_view()
    if arguments.vlm:
        vlm.draw(show_kwargs=dict())