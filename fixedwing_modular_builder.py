import math
import warnings
import pandas as pd
import aerosandbox as asb
import aerosandbox.numpy as np
import neuralfoil as nf
import json
import argparse
from simlib import *
import os
import ast
import binascii

def crc32_update(chunk: str, crc: int = 0) -> int:
    return binascii.crc32(chunk.encode(), crc) & 0xffffffff

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
    parser.add_argument('--altitude', type=float, help="VLM altitude")
    parser.add_argument('--airspeed', type=float, help="VLM airspeed")    
    parser.add_argument('--aoa', type=float, help="VLM Angle of attack (deg)")
    parser.add_argument('--test_aoa_range', action='store_true', help="Test AoA range")
    parser.add_argument('--test_altitude_range', action='store_true', help="Test altitude range")
    parser.add_argument('--test_airspeed_range', action='store_true', help="Test airspeed range")

    parser.add_argument('--aoa_range', type=ast.literal_eval, help="AoA range (deg) to test ie: [0,5]")
    parser.add_argument('--altitude_range', type=ast.literal_eval, help="Altitude range to test ie: [0,1000]")
    parser.add_argument('--airspeed_range', type=ast.literal_eval, help="Airspeed range (m/s) to test ie: [10,50]")
    parser.add_argument('--aoa_increment', type=float, default=1.0, help="AoA range increment")   
    parser.add_argument('--altitude_increment', type=float, default=10.0, help="Altitude increment")   
    parser.add_argument('--airspeed_increment', type=float, default=1.0, help="m/s increment")   
    parser.add_argument('--run_vlm', action='store_true', help="Show 3-view")
    parser.add_argument('--view', action='store_true', help="Show 3-view")
    parser.add_argument('--show_vlm', action='store_true', help="Show vortex lattice simulation")
    parser.add_argument('--export_vsp', action='store_true', help="VSP file export")
    arguments = parser.parse_args()

    #if arguments.run_vlm and not ((arguments.test_aoa_range or arguments.test_airspeed_range or arguments.test_altitude_range) or ):
    #    pass
    test_ranges = (arguments.test_altitude_range or arguments.test_aoa_range or arguments.test_airspeed_range)
    if arguments.run_vlm:
        if (not test_ranges and not (arguments.aoa and arguments.airspeed and arguments.altitude)) \
        or ( arguments.test_altitude_range and not arguments.altitude_range) \
        or ( arguments.test_aoa_range and not arguments.aoa_range) \
        or ( arguments.test_airspeed_range and not arguments.airspeed_range) :
            print("Missing parameters")
            exit()
    if (arguments.aoa and arguments.aoa_range) or (arguments.airspeed and arguments.test_airspeed_range) or (arguments.altitude and arguments.test_altitude_range):
        print("Pick single or range")
        exit()

    # Couple Assembly
    crc = 0
    chasm = assembly_configs[arguments.assembly]
    assembly = []
    for m in chasm["parts"]:
        #print(m)
        assembly.append(m)
        coupling = {"module":"joint-ring-c"}
        assembly.append(coupling)
    assembly.pop()

    canards = chasm["canards"]
    tailless = chasm["tailless"]
    is_vtol = chasm["vtol"]

    base_amps = 1.0


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
        crc = crc32_update(module_name, crc)

        if module_data["type"] == "wingmount":
            wing_spec_name = item_in_assembly["surfaces"]
            wing_data = modules[wing_spec_name]
            num_wings = module_data["wings_n"]
            for zzz in range(num_wings):
                crc = crc32_update(str(wing_data), crc)
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
                #crc = crc32_update(str(item_in_assembly["motors"]), crc)
                #crc = crc32_update(str(item_in_assembly["props"]), crc)
                
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
            batt_wh = (batt_mah/1000.0) * batt_nominal_v

            print(f"{'  ∟ Battery: '+battery_name:<25} | {'in body':<10} | {battery_mass:>9.4f} | {'N/A':>7} | "
                f"{'(at body CG)':>12} | {attached_part_cg_global_x_m:>13.4f} |")
            print(f"  ∟ Capacity: {batt_mah:.0f} mah / {batt_wh:.0f} Wh | Current limit: {batt_curr:.0f} A")

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
        Len  = m["len"]
        R  = m["diam"] / 2

        if m["type"] == "nose":
            add_xsec(x_cursor, 0.0)                                   # tip
            for xi in np.linspace(0, Len, n_cap + 2)[1:-1]:             # skip duplicates
                r = R * math.sqrt(1 - ((xi - Len) / Len) ** 2)
                add_xsec(x_cursor + xi, r)
            add_xsec(x_cursor + Len, R)                                 # joint to tube

        elif m["type"] in ("body", "wingmount"):
            add_xsec(x_cursor, R)
            add_xsec(x_cursor + Len, R)

        elif m["type"] == "tail":
            add_xsec(x_cursor, R)                                     # tube → cap
            for xi in np.linspace(0, Len, n_cap + 2)[1:-1]:
                r = R * math.sqrt(1 - (xi / Len) ** 2)
                add_xsec(x_cursor + xi, r)
            add_xsec(x_cursor + Len, 0.0)                               # tail tip

        x_cursor += Len
        crc = crc32_update(str(x_cursor), crc)

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

    wing_airfoil = None
    tail_airfoil = None

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
                    aoa_deg=wdata["aoa_ang"],
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
    if tail_airfoil is not None:
        S_tail_planform = float(tail_n * tail_span * (tail_chord_root + tail_chord_tip) / 2)
    wing_loading      = float(weight_N / S_main_planform) if S_main_planform > 0 else float('nan')
    wing_to_tail_ratio = float(S_main_planform / S_tail_planform) if S_tail_planform > 0 else float('nan')


    # ==================================================
    # ASSEMBLE AIRPLANE
    # ==================================================
    airplane = asb.Airplane(
        name=chasm["name"],
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
    checksum_hex = f"{crc:08x}"
    print()
    print(str(airplane),checksum_hex)
    print()

    if arguments.export_vsp:
        airplane.export_OpenVSP_vspscript(f"{arguments.assembly}_{checksum_hex}.vspscript")
        #airplane.export_AVL(f'{arguments.export}.avl')
    if arguments.view: 
        airplane.draw_three_view()

    #
    
    # ==================================================
    # THRUST MODEL
    # ==================================================
    prop_diam     = motor_model.prop_diameter * 0.0254
    prop_pitch_m  = motor_model.prop_pitch * 0.0254
    rpm_no_load   = motor_model.kv * batt_S * batt_nominal_v
    V_pitch       = rpm_no_load * prop_pitch_m / 60.0
    static_thrust_g = motor_model.max_thrust * num_motors
    static_thrust_N = (static_thrust_g/1000) * 9.81

    def decay_factor(V):
        """
        Very-rough prop thrust derate versus flight speed.

        - Quadratic fall-off from 100 % at 0 m/s to 30 % at the pitch speed.
        - Linear fade from 30 % to 0 % between 1.0 × and 1.3 × pitch speed.
        - Zero (or treat as drag) beyond 1.3 × pitch speed.
        """
        if V <= V_pitch:                       # 0 - 1 × Vpitch
            slip = 1.0 - (V / V_pitch)
            return 0.30 + 0.70 * slip**2       # 1 → 0.30

        V_zero = 1.3 * V_pitch                # ~where thrust really hits zero
        if V < V_zero:                         # 1.0 - 1.3 × Vpitch
            return 0.30 * (1.0 - (V - V_pitch) / (V_zero - V_pitch))

        return 0.0                             # ≥ 1.3 × Vpitch → no thrust
    
    # ==================================================

    # Wetted area factors (approx. 2 for thin airfoils)
    # For more accuracy: factor = 2 * (1 + t_c_mean + 1.2*(t_c_mean)^2 + ... )
    # Previously the code assumed specific thickness ratios (0.09 for the main
    # wing and 0.08 for the tail). That only works for the stock airfoils. Try
    # to infer t/c from the NACA designation of the chosen airfoil instead.

    # planform & wetted areas
    S_ref  = airplane.s_ref
    S_fuse_wet = fuselage.area_wetted() # More accurate wetted area

    def tc_from_naca(name: str, default: float = 0.1) -> float:
        import re
        m = re.search(r"naca(\d{4,5})", name.lower())
        if m:
            digits = m.group(1)
            try:
                return int(digits[-2:]) / 100.0
            except ValueError:
                pass
        return default

    tc_main = tc_from_naca(wing_airfoil, 0.09)
    if tail_airfoil is not None: tc_tail = tc_from_naca(tail_airfoil, 0.08)

    main_wing_wetted_area_factor = 2.0 * (1 + tc_main + 1.2 * (tc_main ** 2))
    if tail_airfoil is not None: tail_fin_wetted_area_factor = 2.0 * (1 + tc_tail + 1.2 * (tc_tail ** 2))

    S_main_wetted = S_main_planform * main_wing_wetted_area_factor
    if tail_airfoil is not None: S_tail_wetted = S_tail_planform * tail_fin_wetted_area_factor


    # =========================================================================
    # PERFORMANCE ANALYSIS
    # =========================================================================
    if arguments.run_vlm:
        #altitude_m       = arguments.altitude    # [m]
        #airspeed         = arguments.airspeed     # [m/s]
        #aoa_deg          = arguments.aoa

        if arguments.test_altitude_range:
            start_alt = arguments.altitude_range[0]
            end_alt = arguments.altitude_range[1]
            altitude_range = np.arange(start_alt, end_alt + arguments.altitude_increment, arguments.altitude_increment)
        else:
            altitude_range = np.array([arguments.altitude])

        if arguments.test_airspeed_range:
            start_speed = arguments.airspeed_range[0]
            end_speed = arguments.airspeed_range[1]
            airspeed_range = np.arange(start_speed, end_speed + arguments.airspeed_increment, arguments.airspeed_increment)
        else:
            airspeed_range = np.array([arguments.airspeed])

        if arguments.test_aoa_range:
            aoa_start = int(arguments.aoa_range[0])
            aoa_end = int(arguments.aoa_range[1])
            aoa_range = np.arange(aoa_start, aoa_end + arguments.aoa_increment, arguments.aoa_increment)
        else:
            aoa_range = np.array([arguments.aoa])

        performance_results = []

        lastgood = False
        
        last_watts = 0
        best_dat = {"min_speed": 0, "eff": 0, "max_speed":0}
        min_speed_data = {}
        max_speed_data = {}
        vlmdict = {}
        vlmdat = {}

        # ---------- load cache OR generate current-AoA slice only ---------------------
        vlm_file = f"vlm_{arguments.assembly}_{checksum_hex}.json"

        def _json_safe(obj):
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            raise TypeError(f"{type(obj)} not JSON-serialisable")

        # 1) Read any existing cache
        if os.path.isfile(vlm_file):
            with open(vlm_file, "r") as f:
                vlmdict = json.load(f)
            print(f"[cache] loaded VLM results from {vlm_file}")
        else:
            vlmdict = {}

        if test_ranges:
            print("\n" + "="*80)
            print(f"Performance Analysis Ranges")
            print(f"Airspeed: {airspeed_range[0]} to {airspeed_range[len(airspeed_range)-1]}")
            print(f"AoA: {aoa_range[0]} to {aoa_range[len(aoa_range)-1]}")
            print(f"Altitude: {altitude_range[0]} to {altitude_range[len(altitude_range)-1]}")
            print("="*80)
        try:
            for altitude_m in altitude_range:

                for aoa_deg in aoa_range:
                    alpha_rad = math.radians(aoa_deg)
                    print("\n" + "="*80)
                    print(f"Performance Analysis for AoA {aoa_deg:.1f}° at {altitude_m:.1f} m altitude")
                    print("="*80)
                    
                    # Print the header for the live results table
                    print(
                        f"{'Speed (m/s)':<12} | "
                        f"{'Aero L (N)':<14} | "
                        f"{'Aero D (N)':<14} | "
                        f"{'T_req (N)':<16} | "
                        f"{'Max T (N)':<16} | "
                        f"{'Lift %':<14} | "
                        f"{'Motor W':<12} | "
                        f"{'Motor g/W':<12} | "
                        f"{'Throttle (%)':<12} | "
                        f"{'Feasible?':<10}"
                    )
                    print("-" * 145)

                    got_minspeed = False
                    alpha_key = str(int(round(aoa_deg)))
                    alt_key = str(int(round(altitude_m)))
                    if alt_key not in vlmdict:
                        vlmdict[alt_key] = {}

                    if alpha_key in vlmdict[alt_key]:
                        vlmdat = vlmdict[alt_key][alpha_key]               # already cached
                        gen_vlm = False
                    else:
                        vlmdat = {}
                        gen_vlm = True

                    for airspeed in airspeed_range:
                        if airspeed < 0.1:
                            continue
                        else:
                            vlm_results = vlmdat.get(str(airspeed))
                            if not vlm_results:
                                gen_vlm = True
                                #print(f"[cache] AoA {alpha_key}° not found - running VLM sweep for this AoA")
                                op_point = asb.OperatingPoint(
                                    velocity=airspeed, 
                                    alpha=float(aoa_deg)
                                )
                                vlm = asb.VortexLatticeMethod(
                                    airplane=airplane, 
                                    op_point=op_point,
                                    spanwise_resolution=25, 
                                    chordwise_resolution=8,
                                )
                                vlm_results = vlm.run_with_stability_derivatives()
                                vlmdat[str(airspeed)] = vlm.run_with_stability_derivatives()
                                
                                """
                                run_with_stability_derivatives(alpha=True, beta=True, p=True, q=True, r=True)
                                Parameters:
                                alpha (-) - If True, compute the stability derivatives with respect to the angle of attack (alpha).
                                beta (-) - If True, compute the stability derivatives with respect to the sideslip angle (beta).
                                p (-) - If True, compute the stability derivatives with respect to the body-axis roll rate (p).
                                q (-) - If True, compute the stability derivatives with respect to the body-axis pitch rate (q).
                                r (-) - If True, compute the stability derivatives with respect to the body-axis yaw rate (r).
                                """



                        # ==================================================
                        # VORTEX-LATTICE ANALYSIS
                        # ==================================================
                        """
                        'F_g' : an [x, y, z] list of forces in geometry axes [N]
                        'F_b' : an [x, y, z] list of forces in body axes [N]
                        'F_w' : an [x, y, z] list of forces in wind axes [N]
                        'M_g' : an [x, y, z] list of moments about geometry axes [Nm]
                        'M_b' : an [x, y, z] list of moments about body axes [Nm]
                        'M_w' : an [x, y, z] list of moments about wind axes [Nm]
                        'L' : the lift force [N]. Definitionally, this is in wind axes.
                        'Y' : the side force [N]. This is in wind axes.
                        'D' : the drag force [N]. Definitionally, this is in wind axes.
                        'l_b', the rolling moment, in body axes [Nm]. Positive is roll-right.
                        'm_b', the pitching moment, in body axes [Nm]. Positive is pitch-up.
                        'n_b', the yawing moment, in body axes [Nm]. Positive is nose-right.
                        'CL', the lift coefficient [-]. Definitionally, this is in wind axes.
                        'CY', the sideforce coefficient [-]. This is in wind axes.
                        'CD', the drag coefficient [-]. Definitionally, this is in wind axes.
                        'Cl', the rolling coefficient [-], in body axes
                        'Cm', the pitching coefficient [-], in body axes
                        'Cn', the yawing coefficient [-], in body axes

                        Along with additional keys, depending on the value of the alpha, beta, p, q, and r arguments. For example, if alpha=True, then the following additional keys will be present:

                        'CLa', the lift coefficient derivative with respect to alpha [1/rad]
                        'CDa', the drag coefficient derivative with respect to alpha [1/rad]
                        'CYa', the sideforce coefficient derivative with respect to alpha [1/rad]
                        'Cla', the rolling moment coefficient derivative with respect to alpha [1/rad]
                        'Cma', the pitching moment coefficient derivative with respect to alpha [1/rad]
                        'Cna', the yawing moment coefficient derivative with respect to alpha [1/rad]
                        'x_np', the neutral point location in the x direction [m]
                        

                        print()
                        print("VLM Analysis Raw Output")
                        for k, i in vlm_results.items():
                            print(f"{k.rjust(4)} : {i}")
                        print()
                        """

                        # forces & coefficients from VLM
                        LiftN      = float(vlm_results["L"])
                        D_ind  = float(abs(vlm_results["D"])) # Induced drag from VLM
                        My     = float(vlm_results["M_b"][1]) # Pitching moment about airplane.xyz_ref
                        x_cp   = -My / LiftN if LiftN != 0 else float('nan') # Center of pressure relative to xyz_ref

                        CL     = float(vlm_results["CL"])
                        CDi    = float(vlm_results["CD"]) # Induced drag coefficient from VLM
                        Cm     = float(vlm_results["Cm"])

                        # ==================================================
                        # STABILITY DERIVATIVES
                        # ==================================================
                        CLalpha_rad = vlm_results["CLa"] # dCL/dalpha [rad^-1]
                        Cmalpha_rad = vlm_results["Cma"] # dCm/dalpha [rad^-1] about xyz_ref
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
                        #airspeed    = vlm.op_point.velocity
                        # Reynolds number based on main wing root chord for airfoil analysis
                        Re_wing_root = rho * airspeed * wing_chord_root / mu
                        if tail_airfoil is not None:
                            Re_tail_root = rho * airspeed * tail_chord_root / mu

                        # ==================================================
                        # ZERO-LIFT PROFILE DRAG USING NeuralFoil
                        # ==================================================
                        # not zero-lift anymore?
                        CD0_main_airfoil = float(
                            nf.get_aero_from_airfoil(
                                airfoil=asb.Airfoil(wing_airfoil), alpha=aoa_deg, Re=int(Re_wing_root)
                            )["CD"]
                        )
                        if tail_airfoil is not None:
                            CD0_tail_airfoil = float(
                                nf.get_aero_from_airfoil(
                                    airfoil=asb.Airfoil(tail_airfoil), alpha=aoa_deg, Re=int(Re_tail_root)
                                )["CD"]
                            )

                        # flat-plate fuselage drag
                        Re_fuse = rho * airspeed * fuselage_length / mu
                        # Cf_fuse_turbulent = 0.074 / (Re_fuse ** 0.2) # Turbulent flat plate (Schlichting)
                        # Cf_fuse_laminar = 1.328 / (math.sqrt(Re_fuse)) # Laminar flat plate
                        # For mixed flow, a common one: 0.455 / (math.log10(Re_fuse) ** 2.58) (ITTC line)
                        Cf_fuse = 0.455 / (math.log10(Re_fuse) ** 2.58) if Re_fuse > 1e5 else 1.328 / (math.sqrt(Re_fuse))

                        form_factor_fuse = 1 + 60 / ((fuselage_length / tube_diam)**3) + 0.0025 * (fuselage_length / tube_diam) # Hoerner

                        # Component contributions to CD0
                        CD0_components = {
                            "MainWings":  CD0_main_airfoil * S_main_wetted / S_ref,
                            "Fuselage" : Cf_fuse * form_factor_fuse * S_fuse_wet / S_ref
                        }
                        if tail_airfoil is not None: 
                            CD0_components["TailFins"] = CD0_tail_airfoil * S_tail_wetted / S_ref

                        C_D0 = sum(CD0_components.values())

                        # Optional: Add interference drag (e.g., 5-15% of component sum)
                        C_D0_interf_factor = 0.10 # ???????
                        # Problem: This is a crude estimate. Interference drag varies with geometry (e.g., wing-fuselage junctions, motor placement). A flat 10% may over- or underestimate drag, especially for the quadrocket with wingtip motors.
                        C_D0 *= (1 + C_D0_interf_factor)

                        # TODO: propeller drag

                        # ==================================================
                        # CL FROM NeuralFoil
                        # ==================================================
                        CL_max_airfoil_2D = float(nf.get_aero_from_airfoil(
                                airfoil=asb.Airfoil(wing_airfoil),
                                alpha=aoa_deg,
                                Re=int(Re_wing_root),
                            )["CL"])

                        # Estimate 3D CLmax.
                        # Use a simple aspect-ratio based scaling rather than a fixed constant.
                        def clmax_factor_from_AR(AR: float) -> float:
                            if AR <= 0:
                                return 0.8
                            # High AR -> ~0.95, low AR -> ~0.7
                            return max(0.7, min(0.95, AR / (AR + 2.0)))

                        CL_max_3D_factor = clmax_factor_from_AR((airplane.b_ref**2) / airplane.s_ref)
                        CL_max_eff_3D = CL_max_airfoil_2D * CL_max_3D_factor

                        # ==================================================
                        # PERFORMANCE CALCULATIONS
                        # ==================================================
                        # Effective Aspect Ratio of the aircraft
                        AR_eff = (airplane.b_ref**2 / airplane.s_ref) if airplane.s_ref != 0 else float('inf')

                        e_oswald = 0.75  # AI: "Assumed Oswald efficiency factor for X-wing (can range 0.7-0.9)" (TODO: What if it isn't an X-wing? NO ASSUMPTIONS)
                        k_induced_drag = 1 / (math.pi * AR_eff * e_oswald) if AR_eff > 0 and e_oswald > 0 else float('inf')

                        # Total drag coefficient at VLM operating point
                        C_D_total_op_point = C_D0 + CDi # CDi is from VLM at the specific CL (vlm_results["CL"])
                        D_total_op_point   = float(C_D_total_op_point * 0.5 * rho * airspeed ** 2 * S_ref)

                        # Stall speed
                        V_stall   = float(math.sqrt(2 * weight_N / (rho * S_ref * CL_max_eff_3D))) if CL_max_eff_3D > 0 and rho > 0 and S_ref > 0 else float('inf')
                        #print(f"\nCalculated Stall Speed (V_stall): {V_stall:.2f} m/s {V_stall*3.6:.2f} km/h")

                        # L/D at VLM operating point
                        LD_op_point = LiftN / D_total_op_point if D_total_op_point != 0 else float('inf')

                        # Load Factor at VLM operating point
                        load_factor_op_point = LiftN / weight_N if weight_N > 0 else float('nan')

                        q = 0.5 * rho * airspeed**2
                        drag_parasitic_force = q * S_ref * C_D0
                        drag_aero = drag_parasitic_force + D_ind


                        # Net vertical contribution from the wing/fuselage at this α
                        vertical_force_from_aero = (
                            LiftN * math.cos(alpha_rad)     # vertical component of L
                            - drag_aero * math.sin(alpha_rad)   # vertical component of D (always DOWN)
                        )

                        required_vertical_from_thrust = max(0.0, weight_N - vertical_force_from_aero)  # NEVER negative now

                        # Calculate motor performance
                        df = decay_factor(airspeed)
                        avail_t = df * static_thrust_N

                        max_power_h = avail_t * math.cos(alpha_rad)
                        max_power_v = avail_t * math.sin(alpha_rad)
                        lift_proportion = ((vertical_force_from_aero + max_power_v) / weight_N) * 100

                        if is_vtol and abs(math.sin(alpha_rad)) > 1e-9:
                            # REQUIREMENT 1: VERTICAL BALANCE  ( T sin α = ΔW )
                            thrust_req_for_lift_balance = required_vertical_from_thrust / math.sin(alpha_rad)
                        else:
                            thrust_req_for_lift_balance = float("inf")

                        # REQUIREMENT 2: HORIZONTAL BALANCE ( T cos α = D )
                        if abs(math.cos(alpha_rad)) < 1e-9:
                            thrust_req_for_drag_balance = float("inf")
                        else:
                            thrust_req_for_drag_balance = drag_aero / math.cos(alpha_rad) #?
                        
                        
                        # ---------------------------------------------------------------------
                        # 3. THRUST MAGNITUDE NEEDED - the bigger of the two wins
                        # ---------------------------------------------------------------------
                        if is_vtol:
                            total_thrust_req = thrust_req_for_drag_balance + thrust_req_for_lift_balance
                        else:
                            total_thrust_req = thrust_req_for_drag_balance

                        #print(thrust_req_for_lift_balance, total_thrust_req)

                        TperMotor = ((total_thrust_req / 9.81) * 1000) / num_motors
                        equivalent_static_TperMotor = TperMotor / df if df > 1e-6 else float('inf')

                        mot_watt, mot_curr, mot_gW, mot_throt, mot_rpm = motor_model.motor_perf_thrust(equivalent_static_TperMotor)
                        #print(f"Tg:{i*100:<15}  W: {mot_watt:<15.2f} g/W:{mot_gW:<15.2f} A:{mot_curr:<15.2f} RPM:{mot_rpm:<15.2f}")
                        if mot_watt!=math.inf and not lastgood:
                            lastgood = True
                        #if delta_watts > 0:
                        #    print('kek')

                        # Is this flight point possible?
                        is_feasible = total_thrust_req <= avail_t

                        # ---------------------------------------------------------------------
                        # 4. LIVE OUTPUT PER ITERATION
                        # ---------------------------------------------------------------------
                        print(
                            f"{airspeed:<12.1f} | "
                            f"{LiftN:<14.2f} | "
                            f"{drag_aero:<14.2f} | "
                            f"{total_thrust_req:<16.2f} | "
                            f"{avail_t:<16.2f} | "
                            f"{lift_proportion:<14.1f} | "
                            f"{mot_watt:<12.1f} | "
                            f"{mot_gW:<12.1f} | "
                            f"{mot_throt:<12.1f} | "
                            f"{str(is_feasible):<10}"
                        )

                        # Store results for plotting
                        decaying_t = []
                        for i in airspeed_range:
                            decaying_t.append(static_thrust_N * decay_factor(i))
                        decaying_t = decaying_t[:len(performance_results)]
                        # ==================================================
                        # FINAL SUMMARY AND VISUALISATION
                        # ==================================================
                        perf_df = pd.DataFrame(performance_results)

                        simdata = {
                            "aoa_deg": aoa_deg,
                            "altitude": altitude_m,
                            "airspeed": airspeed, 
                            "thrust_req_N": total_thrust_req, 
                            "avail_thrust": avail_t,
                            "aero_lift_N": LiftN, 
                            "aero_drag_N": drag_aero, 
                            "T_req(Drag)": thrust_req_for_drag_balance, 
                            "lift_%": lift_proportion, 
                            "is_feasible": is_feasible,
                            'rho': rho,
                            'Re_wing_root': Re_wing_root,
                            'Re_tail_root': Re_tail_root,
                            'mot_watt': mot_watt, 
                            'mot_curr': mot_curr, 
                            'mot_gW':mot_gW, 
                            'mot_throt':mot_throt, 
                            'mot_rpm':mot_rpm,
                            'equivalent_static_TperMotor':equivalent_static_TperMotor,
                            'LiftN': LiftN,
                            'D_ind': D_ind,
                            'D_total_op_point': D_total_op_point,
                            'My': My,
                            'S_ref': S_ref,
                            'CL': CL,
                            'CDi': CDi,
                            'C_D_total_op_point': C_D_total_op_point,
                            'Cm': Cm,
                            'CD0_main_airfoil': CD0_main_airfoil,
                            'CD0_tail_airfoil': CD0_tail_airfoil,
                            'Cf_fuse': Cf_fuse,
                            'Re_fuse': Re_fuse,
                            'form_factor_fuse': form_factor_fuse,
                            'CD0_components': CD0_components,
                            'C_D0_interf_factor': C_D0_interf_factor,
                            'C_D0': C_D0,
                            'AR_eff': AR_eff,
                            'e_oswald': e_oswald,
                            'k_induced_drag': k_induced_drag,
                            'LD_op_point': LD_op_point,
                            'static_thrust_N': static_thrust_N,
                            'load_factor_op_point': load_factor_op_point,
                            'CL_max_airfoil_2D': CL_max_airfoil_2D,
                            'CL_max_eff_3D': CL_max_eff_3D,
                            'CL_max_3D_factor': CL_max_3D_factor,
                            'V_stall': V_stall,
                            'x_cp': x_cp,
                            'x_np': x_np,
                            'static_margin_pct': static_margin_pct,
                            'CLalpha_deg': CLalpha_deg,
                            'Cmalpha_deg': Cmalpha_deg,
                        }
                        if is_feasible and mot_watt!=math.inf and airspeed>best_dat['max_speed']:
                            best_dat['max_speed'] = airspeed
                            max_speed_data = simdata

                        if lift_proportion >= 100 and not got_minspeed:
                            if airspeed > best_dat['min_speed'] and mot_gW > best_dat['eff']:
                                min_speed_data = simdata
                                best_dat['min_speed'] = airspeed
                                best_dat['eff'] = mot_gW
                                print('New best ----------------')
                            got_minspeed = True


                        if lastgood and mot_watt == math.inf:
                            print("INFO: Exiting loop after second motor performance failure.")
                            break
                    
                    if gen_vlm:
                        vlmdict[alt_key][alpha_key] = vlmdat               # add new slice
                        with open(vlm_file, "w") as f:
                            json.dump(vlmdict, f, default=_json_safe, indent=2)
                        print(f"[cache] Alt {alt_key}m AoA {alpha_key}° saved to {vlm_file}")

                    if not max_speed_data:
                        max_speed_data = simdata
                    if not min_speed_data:
                        min_speed_data = simdata
        except KeyboardInterrupt:
            print("Run aborted")

        print("\n--- Analysis Complete ---")
        """
        # Find the minimum thrust required and the speed at which it occurs (best endurance point)
        if not perf_df.empty:
            min_thrust_point = perf_df.loc[perf_df['thrust_req_N'].idxmin()]
            print(f"Minimum Thrust Required: {min_thrust_point['thrust_req_N']:.2f} N at {min_thrust_point['airspeed_ms']:.1f} m/s (Best Endurance Speed).")

        try:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            fig.suptitle(f'VTOL Performance Breakdown at {aoa_deg}° AoA', fontsize=16)

            # Plot 1: Thrust Required vs. Thrust Available
            ax1.plot(perf_df['airspeed_ms'], perf_df['thrust_req_N'], 'b-', label='Total Motor Thrust Required', linewidth=2)
            ax1.plot(perf_df['airspeed_ms'], decaying_t, color='green', label='Thrust Available', linewidth=2)
            ax1.set_xlabel('Airspeed [m/s]')
            ax1.set_ylabel('Thrust [N]')
            ax1.grid(True, which='both', linestyle=':')
            ax1.legend()
            ax1.set_ylim(bottom=0)

            # Plot 2: Lift Contribution Breakdown
            ax2.plot(perf_df['airspeed_ms'], perf_df['motor_lift_%'], 'r-', label='% of Vertical Lift from Motors', linewidth=2)
            ax2.axhline(0, color='black', linestyle=':', linewidth=1) # Mark the crossover point
            ax2.set_xlabel('Airspeed [m/s]')
            ax2.set_ylabel('Contribution to Vertical Lift [%]')
            ax2.grid(True, which='both', linestyle=':')
            ax2.legend()
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

        except (ImportError, ValueError) as e:
            print(f"\nPlot generation failed. Error: {e}")
        """

        tot_watt_rng = min_speed_data['mot_watt'] * num_motors
        batt_t_rng = batt_wh / tot_watt_rng * 60
        tot_curr_rng = min_speed_data['mot_curr'] * num_motors
        batt_t_rng = batt_mah / (tot_curr_rng*1000) * 60
        flt_range_rng = (min_speed_data['airspeed'] * (batt_t_rng*60)) / 1000.0

    # ==================================================
    # RESULTS
    # ==================================================
    if arguments.run_vlm:
        simdata = max_speed_data

    print("\n" + "="*40)
    print("OVERALL SUMMARY:")
    print("="*40)
    print(f"Wing AoA         : {wing_aoa:.3f} deg")
    if tail_airfoil is not None:
        print(f"Tail AoA         : {tail_aoa:.3f} deg")
    print(f"Total Fuselage Length : {fuselage_length:.4f} m")
    print(f"Total Mass            : {total_mass_kg:.4f} kg")
    print(f"Total N Mass          : {weight_N:.4f} N")
    print(f"Total Moment (kg*m)   : {total_moment_kg_m:.4f} kg*m")
    print(f"Center of Gravity (CG_x): {cg_x_m:.4f} m (from fuselage front, x=0)")
    print()
    print(f"--- Motors ---")
    print(f"Motor name       : {brand} {model} {kv}KV")
    print(f"Motor num        : {num_motors}")
    print(f"Propellers       : {prop}")
    print(f"Total max thrust : {static_thrust_g:.2f} g")
    print()
    print(f"--- Battery ---")
    print(f"Name       : {battery_name}")
    print(f"Type       : {batt_S}S{batt_P}P")
    print(f"Weight     : {battery_mass} kg")
    print(f"Capacity   : {batt_mah:.2f} mAh {batt_wh:.2f} Wh")
    print()
    print(f"--- Areas ---")
    print(f"Reference Area (S_ref)                   : {S_ref:.4f} m² (Total main wing planform)")
    print(f"Main Wing Planform Area (S_main)         : {S_main_planform:.4f} m²")
    if tail_airfoil is not None:
        print(f"Tail Fin Planform Area (S_tail)          : {S_tail_planform:.4f} m²")
    print(f"Fuselage Wetted Area (S_fuse_wet)        : {S_fuse_wet:.4f} m²")
    print(f"Main Wing Wetted Area (S_main_wet)       : {S_main_wetted:.4f} m²")
    if tail_airfoil is not None:
        print(f"Tail Fin Wetted Area (S_tail_wet)        : {S_tail_wetted:.4f} m²")
    print(f"Wing loading (W/S_main)                  : {wing_loading:.2f} N/m²")
    print(f"Wing-to-tail area ratio (S_main/S_tail)  : {wing_to_tail_ratio:.2f}")
    print()


    if arguments.run_vlm:

        print("\n" + "="*40)
        print("VLM RESULTS:")
        print("="*40)
        if not simdata["is_feasible"]:
            print()
            print("="*10,"Invalid motor data", "="*10)
            print()
        print(f"Airspeed                 : {simdata['airspeed']:.1f} m/s")
        print(f"Altitude                 : {simdata['altitude']:.1f} m")
        print(f"Atm density              : {simdata['rho']:.3f} kg/m³")
        print(f"Reynolds (Main Wing Root): {simdata['Re_wing_root']:.2e}")
        if tail_airfoil is not None:
            print(f"Reynolds (Tail Fin Root) : {simdata['Re_tail_root']:.2e}")
        print()
        print(f"--- Lift and Drag ---")
        print(f"Lift (L)                                 : {simdata['LiftN']:.2f} N")
        print(f"Induced drag (Dᵢ)                        : {simdata['D_ind']:.2f} N")
        print(f"Total drag at Op Point (Dₜ_op)           : {simdata['D_total_op_point']:.2f} N")
        print(f"Pitching moment (Mᵧ about xyz_ref)       : {simdata['My']:.2f} N·m")
        print(f"Effective Aspect Ratio (AR_eff)          : {simdata['AR_eff']:.2f}")
        print(f"Assumed Oswald Efficiency (e)            : {simdata['e_oswald']:.2f}")
        print(f"Induced Drag Factor (k = 1/(π*AR*e))     : {simdata['k_induced_drag']:.4f}")
        print(f"L/D at VLM Op Point                      : {simdata['LD_op_point']:.2f}")
        print(f"Lift coefficient (C_L)                   : {simdata['CL']:.4f}")
        print(f"Induced-drag coefficient (C_Dᵢ)          : {simdata['CDi']:.4f}")
        print(f"Total drag coefficient (C_Dₜ_op)         : {simdata['C_D_total_op_point']:.4f}")
        print(f"Pitching-moment coefficient (C_m)        : {simdata['Cm']:.4f}")
        print()
        print(f"--- Zero-Lift Drag (C_D0) Breakdown ---")
        print(f"Main wing airfoil ({wing_airfoil}) CD @ Re={simdata['Re_wing_root']:.1e}, α=0°: {simdata['CD0_main_airfoil']:.5f}")
        if tail_airfoil is not None:
            print(f"Tail fin airfoil ({tail_airfoil}) CD @ Re={simdata['Re_tail_root']:.1e}, α=0°:  {simdata['CD0_tail_airfoil']:.5f}")
        print(f"Fuselage skin friction coeff (Cf_fuse)   : {simdata['Cf_fuse']:.5f} (Re_fuse={simdata['Re_fuse']:.1e})")
        print(f"Fuselage form factor (FF_fuse)           : {simdata['form_factor_fuse']:.3f}")
        print(f"CD0 Contribution - Main Wings            : {simdata['CD0_components']['MainWings']:.5f} ({(simdata['CD0_components']['MainWings']/simdata['C_D0']*100 if simdata['C_D0']!=0 else 0):.1f}%)")
        if tail_airfoil is not None:
            print(f"CD0 Contribution - Tail Fins             : {simdata['CD0_components']['TailFins']:.5f} ({(simdata['CD0_components']['TailFins']/simdata['C_D0']*100 if simdata['C_D0']!=0 else 0):.1f}%)")
        print(f"CD0 Contribution - Fuselage              : {simdata['CD0_components']['Fuselage']:.5f} ({(simdata['CD0_components']['Fuselage']/simdata['C_D0']*100 if simdata['C_D0']!=0 else 0):.1f}%)")
        print(f"Interference Drag Factor Applied         : {simdata['C_D0_interf_factor']*100:.1f}%")
        print(f"Total Zero-lift drag coeff (C_D₀)        : {simdata['C_D0']:.5f}")
        print()
        #print(f"--- Performance Estimates ---")
        #print(f"TotalVForce (Lift + ThrustY)             : {simdata['TotalVForce']:.2f} N")
        #print(f"TotalVForce/Weight                       : {simdata['TLW']:.2f}")
        """print(f"  Best Glide Ratio (L/D_max)             : {simdata['best_glide_ratio']:.2f}")
        print(f"  Best Glide Angle                       : {math.degrees(simdata['best_glide_angle_rad']) if not math.isnan(simdata['best_glide_angle_rad']) else 'NaN':.2f} deg")
        print(f"  Airspeed for Best Glide (V_bg)         : {simdata['V_at_best_glide']:.2f} m/s")
        print(f"  Sink Rate at Best Glide                : {simdata['sink_rate_at_best_glide']:.2f} m/s")
        print(f"Est. Min Sink Rate (parabolic approx.)   : {simdata['min_sink_rate_approx']:.2f} m/s")
        print(f"  Airspeed for Min Sink (V_ms)         : {simdata['V_at_min_sink_approx']:.2f} m/s (at CL ≈ {simdata['CL_min_sink_approx']:.3f})")
        print()"""
        
        print("--- Max range speed ---")
        print(f"AoA          : {min_speed_data['aoa_deg']:.1f} deg")
        print(f"Altitude     : {min_speed_data['altitude']:.1f} m")
        print(f"Airspeed     : {min_speed_data['airspeed']:.2f} m/s {min_speed_data['airspeed']*3.6:.2f} km/h")
        #print(f"P_air {simdata['P_air_rng']:.2f} W")
        print(f"Throttle     : {min_speed_data['mot_throt']:.1f} %")
        print(f"Watts        : {tot_watt_rng:.2f} W")
        print(f"Current      : {tot_curr_rng:.2f} A")
        #print(f"I_total {min_speed_data['tot_mot_curr']:.2f} A")
        print(f"g/W          : {min_speed_data['mot_gW']:.2f}")
        print(f"Flight time  : {batt_t_rng:.2f} min ({batt_t_rng/60.0:.2f} h)")
        print(f"Flight range : {flt_range_rng:.2f} km")
        print()
        #print("-"*106)
        print(f"--- Performance at VLM Operating Point (V={simdata['airspeed']:.1f}m/s, T_static={simdata['static_thrust_N']:.1f}N) ---")
        print(f"Load Factor (L/W) at Op Point            : {simdata['load_factor_op_point']:.2f}")
        print(f"Main wing airfoil 2D CL ({wing_airfoil:<8})      : {simdata['CL_max_airfoil_2D']:.4f} (Re={simdata['Re_wing_root']:.1e})")
        print(f"Estimated 3D CL                          : {simdata['CL_max_eff_3D']:.4f} (using factor {simdata['CL_max_3D_factor']:.2f} on 2D C_Lmax)")
        print(f"Stall speed (Vₛ at W={weight_N:<5.1f}N)             : {simdata['V_stall']:.2f} m/s {V_stall*3.6:.2f} km/h")
        print()
        print(f"--- Stability & Control (at VLM Op Point) ---")
        print(f"CG Location (x_cg from nose)             : {cg_x_m:.3f} m")
        print(f"Center of pressure (x_cp from xyz_ref)   : {simdata['x_cp']:.3f} m")
        print(f"Neutral Point (x_np from nose)           : {simdata['x_np']:.3f} m")
        print(f"Static Margin (SM)                       : {simdata['static_margin_pct']:.1f} %c_ref")
        print(f"Lift curve slope (CLα)                   : {simdata['CLalpha_deg']:.4f} /deg")
        print(f"Moment curve slope (Cmα about xyz_ref)   : {simdata['Cmalpha_deg']:.4f} /deg")
        print()


    # ==================================================
    # VISUALISATION
    # ==================================================
    

    if arguments.show_vlm:
        vlm.draw(show_kwargs=dict())