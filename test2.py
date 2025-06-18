# [ ... All code from the beginning of the file up to this point remains IDENTICAL ... ]

import pandas as pd
import math
import numpy as np
import itertools
from matplotlib import pyplot as plt 

class MotorData:
    def __init__(self, brand, motor, prop, kv, serie):
        self.brand = brand
        self.motor = motor
        self.prop = prop
        self.kv = int(kv)
        self.serie = int(serie)

        motor_data = pd.read_csv('motordata.csv')
        group = motor_data[(motor_data['Brand'] == self.brand) & 
                           (motor_data['Motor'] == self.motor) & 
                           (motor_data['Prop'] == self.prop) & 
                           (motor_data['KV'] == self.kv) & 
                           (motor_data['Data Batt S'] == self.serie)]

        if group.empty:
            print(f"No data found for {brand} {motor} {prop} {kv} {serie}S")
            exit(0)
            
        self.throttle = group['T'].str.rstrip('%').astype(float).values
        self.thrust = group['Thrust g'].values
        self.max_thrust = self.thrust[len(self.thrust)-1]
        self.current = group['A'].values
        self.max_draw = self.current[len(self.current)-1]
        self.rpm = group['RPM'].values
        self.watts = group['W'].values
        self.motor_batt_s = group['Data Batt S'].iloc[0]
        self.efficiency = group['Efficiency'].values
        self.motor_weight = group['Weight g'].iloc[0]
        self.stator_diameter = group['Stator Diameter'].iloc[0]
        self.stator_height = group['Stator height'].iloc[0]
        self.kv = group['KV'].iloc[0]
        self.prop_diameter = group['Data P size'].iloc[0]
        self.prop_pitch = group['Data P pitch'].iloc[0]

        thrust_values = np.arange(50, self.max_thrust, 10)
        for i in ['throttle','current','rpm','watts','efficiency']:
            self.plot_thrust_to(i, thrust_values)

    # Interpolation function for each variable
    def thrust_to_throttle(self, thrust_values):
        return np.interp(thrust_values, self.thrust, self.throttle)

    def thrust_to_current(self, thrust_values):
        return np.interp(thrust_values, self.thrust, self.current)

    def thrust_to_rpm(self, thrust_values):
        return np.interp(thrust_values, self.thrust, self.rpm)

    def thrust_to_watts(self, thrust_values):
        return np.interp(thrust_values, self.thrust, self.watts)

    def thrust_to_efficiency(self, thrust_values):
        return np.interp(thrust_values, self.thrust, self.efficiency)

    # Example plot function
    def plot_thrust_to(self, target_variable, thrust_values):
        true_values = np.interp(thrust_values, self.thrust, getattr(self, target_variable))
        predicted_values = getattr(self, f'thrust_to_{target_variable}')(thrust_values)

        # Plot true vs predicted values
        #plt.figure(figsize=(8, 6))
        #plt.plot(thrust_values, true_values, 'o-', label='True Values', color='blue')
        #plt.plot(thrust_values, predicted_values, 'x-', label='Interpolated Values', color='red')
        #plt.title(f'Interpolated vs True Values for {target_variable} ({self.brand} {self.motor} {self.prop})')
        #plt.xlabel('Thrust (g)')
        #plt.ylabel(target_variable.capitalize())
        #plt.legend()
        #plt.grid(True)
        #plt.savefig(f'{brand} {motor} {prop} {target_variable}.png')
        #plt.show()


    def motor_perf_thrust(self, val): # thrust g
        # Note: The global keyword here is an anti-pattern, but we keep it
        # to match the original code's structure.
        global number_of_motors
        if val > self.max_thrust:
            # Return specific values indicating failure instead of None
            return float('nan'), float('nan'), float('nan'), float('nan')
        
        # Check if requested thrust is below the minimum measured thrust
        if val < self.thrust[0]:
             # Linearly extrapolate down to zero thrust from the first data point
            ratio = val / self.thrust[0]
            power_per_motor = self.watts[0] * ratio
            current_per_motor = self.current[0] * ratio
            eff = self.efficiency[0] * ratio # Efficiency should also go to 0, this is a reasonable approx
            throttle = self.throttle[0] * ratio
        else:
            power_per_motor = self.thrust_to_watts(val)
            current_per_motor = self.thrust_to_current(val)
            eff = self.thrust_to_efficiency(val)
            throttle = self.thrust_to_throttle(val)

        return power_per_motor, current_per_motor, eff, throttle

class LiIonBatteryData:    
    def __init__(self, brand, model, cellformat, serie, parallel):
        self.brand = brand
        self.model = model
        self.format = int(cellformat)
        self.parallel = int(parallel)
        self.serie = int(serie)
        self.max_v = 4.2
        self.nominal_v = 3.6
        self.min_v = 2.5

        batt_data = pd.read_csv('batterydata.csv')
        group = batt_data[(batt_data['Brand'] == self.brand) & 
                           (batt_data['Name'] == self.model) & 
                           (batt_data['Format'] == self.format)]

        if group.empty:
            print(f"No data found for {brand} {model} {cellformat}")
            exit(0)
        self.cell_weight = group['Weight'].iloc[0] / 1000.0
        self.cell_c = group['C-rating'].iloc[0]
        self.cell_cap_a = group['CapacityA'].iloc[0]
        self.cell_cap_wh = group['CapacityWh'].iloc[0]
        self.pack_weight = self.cell_weight * serie * parallel
        self.pack_amps = self.cell_cap_a * parallel
        self.pack_wh = self.cell_cap_wh * parallel
        self.pack_current = self.cell_c * self.pack_amps
        cell_format_str = str(self.format) # Convert to string for comparison
        self.cell_len = 0.070 if cell_format_str == "21700" else 0.065
        self.cell_width = 0.021 if cell_format_str == "21700" else 0.018
        self.pack_len = self.cell_len * self.parallel
        self.pack_width = -1

import math
import warnings
import pandas as pd
import aerosandbox as asb
import aerosandbox.numpy as np
import neuralfoil as nf
import json
import argparse
from scipy.optimize import brentq


# Silence NumPy scalar-conversion deprecation spam
warnings.filterwarnings("ignore", category=DeprecationWarning)

def create_elliptical_fuselage(
    length: float,
    diameter: float,
    cap_len: float,
    n_cap: int = 12,
    n_cyl: int = 8,
) -> asb.Fuselage:
    r = diameter / 2
    x_cyl0, x_cyl1 = cap_len, length - cap_len
    xsecs = []

    # nose cap
    for xi in np.linspace(0, cap_len, n_cap):
        radius = r * math.sqrt(max(0.0, 1 - ((xi - cap_len) / cap_len) ** 2))
        xsecs.append(asb.FuselageXSec(xyz_c=[xi, 0, 0], radius=radius))

    # straight cylinder
    for xi in np.linspace(x_cyl0, x_cyl1, n_cyl):
        xsecs.append(asb.FuselageXSec(xyz_c=[xi, 0, 0], radius=r))

    # tail cap
    for xi in np.linspace(0, cap_len, n_cap):
        radius = r * math.sqrt(max(0.0, 1 - (xi / cap_len) ** 2))
        xsecs.append(asb.FuselageXSec(xyz_c=[x_cyl1 + xi, 0, 0], radius=radius))

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
        twist=aoa_deg,
        airfoil=base_af,
        control_surface_is_symmetric=False,
        control_surface_deflection=0.0,
    )
    tip = asb.WingXSec(
        xyz_le=[x_le+dx_sweep, y_tip, z_tip],
        chord=chord_tip,
        twist=aoa_deg,
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
    parser.add_argument('--airspeed', type=float, default=30, help="Sim airspeed for initial op-point analysis")
    parser.add_argument('--aoa', type=float, default=0.0, help="Angle of attack for initial op-point analysis")
    parser.add_argument('--view', default=True, action='store_true', help="Show 3-view")
    parser.add_argument('--vlm', action='store_true', help="Show vortex lattice simulation")
    parser.add_argument('--export', default=None, help="VSP file export filename")
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

    # This represents the current draw of the avionics, flight controller, etc.
    # Should ideally be a configurable parameter.
    base_amps = 1.0

    # ==================================================
    # SIMULATION PARAMETERS
    # ==================================================
    altitude_m       = arguments.altitude    # [m]
    airspeed_op_point = arguments.airspeed     # [m/s]
    aoa_deg_op_point  = arguments.aoa

    # ==================================================
    # ASSEMBLY & MASS/LENGTH/CG SUMMARY
    # ==================================================
    wing_sections        = []
    fuselage_length    = 0.0
    total_mass_kg        = 0.0
    total_moment_kg_m    = 0.0  # Sum of (mass_i * cg_x_i)
    current_x_fuselage_m = 0.0  # Tracks the x-coordinate of the FRONT of the current module
    tube_diam = 0.0

    print("="*120)
    print(f"Airframe: {arguments.assembly}")
    print("="*120)
    header = (f"{'Module/Part Name':<25} | {'Type':<10} | {'Mass (kg)':>9} | {'Len (m)':>7} | "
            f"{'Local CG (m)':>12} | {'Global CG (m)':>13} | {'Mod Front_x (m)':>14}")
    print(header)
    print("-" * len(header))
    num_motors = 0
    
    # This is a known issue from the original code. It assumes one motor type for the whole
    # aircraft and will be overwritten if multiple motor types are defined. The refactoring
    # works with this limitation.
    motor_model = None

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
                # TODO: This assumes the battery S-count is known. A better approach would be to
                # determine battery details first, then initialize the motor. Here we hardcode to 6S.
                batt_s_for_motor = 6 
                motor_model = MotorData(brand, model, prop, kv, batt_s_for_motor)
                num_motors_per_wing = wing_data["motor_n"]
                num_motors += num_wings * num_motors_per_wing # Use += to accumulate
                motor_mass_each = motor_model.motor_weight  / 1000.0
                total_wing_motors_mass = (num_wings * num_motors_per_wing) * motor_mass_each
                
                print(f"{'    ∟ Motor: '+motor_name:<25} | {'on wing':<10} | {total_wing_motors_mass:>9.4f} | {'N/A':>7}"
                    f"{'(at mount CG)':>12} | {attached_part_cg_global_x_m:>13.f}")
                print(f"{'    ∟ Propeller: '+prop:<25} ")

                total_mass_kg     += total_wing_motors_mass
                total_moment_kg_m += total_wing_motors_mass * attached_part_cg_global_x_m


        if module_data["type"] == "body" and module_data.get("payload", False):
            payload_mass = assembly[i]["payload_mass"]

            print(f"{'  ∟ Payload: ':<25} | {'in body':<10} | {payload_mass:>9.4f} | {'N/A':>7} | "
                f"{'(at body CG)':>12} | {attached_part_cg_global_x_m:>13.4f} |")

            total_mass_kg     += payload_mass
            total_moment_kg_m += payload_mass * attached_part_cg_global_x_m

        if module_data["type"] == "body" and module_data.get("battery", False):
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

        if module_data["type"] == "tail" and module_data.get("motor", False):
            motor_name = item_in_assembly["motors"]
            prop = item_in_assembly["props"]
            brand, model, kv = motor_name.split("_")
            # TODO: This assumes the battery S-count is known. A better approach would be to
            # determine battery details first, then initialize the motor. Here we hardcode to 6S.
            batt_s_for_motor = 6
            motor_model = MotorData(brand, model, prop, int(kv), batt_s_for_motor)
            motors_in_tail = module_data["motor_n"]
            num_motors += motors_in_tail # Use += to accumulate
            motor_mass_each = motor_model.motor_weight / 1000.0
            total_tail_motors_mass = motors_in_tail * motor_mass_each

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
    static_thrust_g = motor_model.max_thrust * num_motors
    static_thrust_N = (static_thrust_g / 1000.0) * 9.81

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
    x_offset = 0.0
    xsecs = []
    n_cap = 12

    for mn in assembly:
        m = modules[mn["module"]]
        length = m["len"]
        r      = m["diam"] / 2

        if m["type"] == "nose":
            # Elliptical nose cap
            for xi in np.linspace(0, length, n_cap):
                radius = r * math.sqrt(max(0.0, 1 - ((xi - length) / length) ** 2))
                xsecs.append(
                    asb.FuselageXSec(
                        xyz_c=[x_offset + xi, 0, 0],
                        radius=radius
                    )
                )
            x_offset += length

        elif m["type"] in ["body", "wingmount"]:
            # Straight cylinder segment
            xsecs.append(asb.FuselageXSec(xyz_c=[x_offset,        0, 0], radius=r))
            xsecs.append(asb.FuselageXSec(xyz_c=[x_offset+length, 0, 0], radius=r))
            x_offset += length

        elif m["type"] == "tail":
            # Cylinder base
            xsecs.append(asb.FuselageXSec(xyz_c=[x_offset, 0, 0], radius=r))
            # Elliptical tail cap
            for xi in np.linspace(0, length, n_cap):
                radius = r * math.sqrt(max(0.0, 1 - (xi / length) ** 2))
                xsecs.append(
                    asb.FuselageXSec(
                        xyz_c=[x_offset + xi, 0, 0],
                        radius=radius
                    )
                )
            x_offset += length

    fuselage = asb.Fuselage(
        name="Fuselage",
        xsecs=xsecs
    )
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
    # print("X pos map:", module_x_map)
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
                # print(n_wing_modules, wdata)
                wingang = m["wings_ang"] * flip
                
                x_pos = module_x_map[i] - (wdata["chord_root"]/2)
                y_pos = (m["diam"] / 2) * math.cos(math.radians(wingang))
                z_pos = (m["diam"] / 2) * math.sin(math.radians(wingang))
                wing = create_surface(
                    idx=j,
                    span=wdata["span"] + fuselage_radius,
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

    # ==================================================
    # VORTEX-LATTICE ANALYSIS
    # ==================================================
    vlm = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(velocity=airspeed_op_point, alpha=aoa_deg_op_point),
        spanwise_resolution=25, chordwise_resolution=8
    )
    aero_op_point = vlm.run()

    print()
    print(f"VLM Analysis Raw Output (V={airspeed_op_point} m/s, alpha={aoa_deg_op_point} deg)")
    for k, v in aero_op_point.items():
        print(f"{k.rjust(4)} : {v}")
    print()

    # forces & coefficients from VLM at the specified operating point
    L_op     = float(aero_op_point["L"])
    D_ind_op = float(abs(aero_op_point["D"])) # Induced drag from VLM
    My_op    = float(aero_op_point["M_b"][1]) # Pitching moment about airplane.xyz_ref
    x_cp_op  = -My_op / L_op if L_op != 0 else float('nan') # Center of pressure relative to xyz_ref

    CL_op    = float(aero_op_point["CL"])
    CDi_op   = float(aero_op_point["CD"]) # Induced drag coefficient from VLM
    Cm_op    = float(aero_op_point["Cm"])

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
    
    # Reynolds number based on main wing root chord for airfoil analysis
    Re_wing_root = rho * airspeed_op_point * wing_chord_root / mu
    Re_tail_root = rho * airspeed_op_point * tail_chord_root / mu

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
    S_fuse_wet = fuselage.area_wetted()

    main_wing_wetted_area_factor = 2.02 # Approximation for typical airfoils (t/c ~ 9-12%)
    tail_fin_wetted_area_factor = 2.016  # Approximation for thinner airfoils (t/c ~ 8%)

    S_main_wetted = S_main_planform * main_wing_wetted_area_factor
    S_tail_wetted = S_tail_planform * tail_fin_wetted_area_factor

    Re_fuse = rho * airspeed_op_point * fuselage_length / mu
    Cf_fuse = 0.455 / (math.log10(Re_fuse) ** 2.58) if Re_fuse > 1e5 else 1.328 / (math.sqrt(Re_fuse))
    form_factor_fuse = 1 + 60 / ((fuselage_length / tube_diam)**3) + 0.0025 * (fuselage_length / tube_diam) # Hoerner
    CD0_contrib_fuse = Cf_fuse * form_factor_fuse * S_fuse_wet / S_ref

    CD0_contrib_main = CD0_main_airfoil * S_main_wetted / S_ref
    CD0_contrib_tail = CD0_tail_airfoil * S_tail_wetted / S_ref

    C_D0 = CD0_contrib_main + CD0_contrib_tail + CD0_contrib_fuse
    C_D0_interf_factor = 0.10 # Estimate for interference drag
    C_D0 *= (1 + C_D0_interf_factor)

    # ==================================================
    # 3D LIFT AND DRAG CHARACTERISTICS
    # ==================================================
    AR_eff = (airplane.b_ref**2 / S_ref) if S_ref != 0 else float('inf')

    # Calculate Oswald efficiency factor from the VLM op_point results
    # e = CL^2 / (pi * AR * CDi)
    if CDi_op > 1e-6: # Avoid division by zero
        e_oswald = CL_op**2 / (math.pi * AR_eff * CDi_op)
        e_oswald = np.clip(e_oswald, 0.5, 1.0) # Cap to reasonable values
    else:
        e_oswald = 0.8 # Fallback for cases with no induced drag (e.g., alpha=0)
    
    k_induced_drag = 1 / (math.pi * AR_eff * e_oswald)

    # Estimate 3D C_Lmax using a more accurate method (Helmbold-Prandtl)
    alphas_clmax_sweep = np.linspace(-5, 20, 51)
    CLs_airfoil = np.array([
        nf.get_aero_from_airfoil(airfoil=asb.Airfoil(wing_airfoil), alpha=float(a), Re=int(Re_wing_root))["CL"]
        for a in alphas_clmax_sweep
    ])
    CL_max_airfoil_2D = float(CLs_airfoil.max())
    alpha_stall_airfoil_2D = float(alphas_clmax_sweep[CLs_airfoil.argmax()])
    CL_max_3D = CL_max_airfoil_2D * (AR_eff / (AR_eff + 2 * (AR_eff + 4) / (AR_eff + 2))) # Helmbold approximation

    # ==================================================
    # PERFORMANCE CALCULATION (NEW METHODOLOGY)
    # ==================================================
    print("\n" + "="*50)
    print("STARTING FULL PERFORMANCE SWEEP ANALYSIS...")
    print("="*50)
    
    # Stall speed is the absolute minimum speed for level flight
    V_stall = math.sqrt(2 * weight_N / (rho * S_ref * CL_max_3D)) if CL_max_3D > 0 else float('inf')

    # Effective pitch speed of the propeller
    # Use max RPM from motor data for a more realistic estimate than no-load RPM
    prop_pitch_m = motor_model.prop_pitch * 0.0254
    rpm_at_max_thrust = motor_model.rpm[-1]
    V_pitch_effective = rpm_at_max_thrust * prop_pitch_m / 60.0

    def get_thrust_available(V, V_pitch, T_static, num_motors):
        """ A simple but more realistic model of thrust fall-off with airspeed. """
        if V >= V_pitch:
            return 0.0
        # Parabolic fall-off model
        return T_static * (1 - (V / V_pitch)**2)
    
    def get_lift_drag_at_V(airplane, V, alpha_deg, op_point_base):
        """ Helper to run VLM for a specific V and alpha. """
        op_point_current = asb.OperatingPoint(velocity=V, alpha=alpha_deg)
        vlm = asb.VortexLatticeMethod(airplane=airplane, op_point=op_point_current)
        aero = vlm.run()
        return float(aero['L']), float(aero['CD'])

    def find_level_flight_alpha(alpha_deg, airplane, V, weight_N, op_point_base):
        """ Function for the root-finder. Returns L-W. """
        L, _ = get_lift_drag_at_V(airplane, V, alpha_deg, op_point_base)
        return L - weight_N

    # --- Performance Sweep ---
    results = []
    # Sweep from just above stall speed up to effective pitch speed
    airspeeds_to_test = np.linspace(V_stall * 1.05, V_pitch_effective * 0.98, 50)
    
    op_point_base = asb.OperatingPoint()

    for V in airspeeds_to_test:
        try:
            # Find the alpha for level flight (L=W) at this speed
            # Search for alpha between -5 and stall alpha + 5 degrees
            alpha_level_flight_deg = brentq(
                f=find_level_flight_alpha,
                a=-5,
                b=alpha_stall_airfoil_2D + 5,
                args=(airplane, V, weight_N, op_point_base)
            )

            # Now that we have the alpha, get the total drag at this flight condition
            q = 0.5 * rho * V**2
            _, CDi_level = get_lift_drag_at_V(airplane, V, alpha_level_flight_deg, op_point_base)
            CD_total_level = C_D0 + CDi_level
            
            thrust_required_N = CD_total_level * q * S_ref
            thrust_available_N = get_thrust_available(V, V_pitch_effective, static_thrust_N, num_motors)

            # If we need more thrust than is available, we can't fly at this speed
            if thrust_required_N > thrust_available_N:
                continue # Skip to next speed

            thrust_per_motor_g = (thrust_required_N / num_motors / 9.81) * 1000

            # Get motor performance for the required thrust
            P_m, I_m, gW, thr = motor_model.motor_perf_thrust(thrust_per_motor_g)
            
            if math.isnan(P_m): # motor_perf_thrust failed
                continue

            I_total = (I_m * num_motors) + base_amps
            P_elec_total = P_m * num_motors + (base_amps * batt_nominal_v * batt_S)

            flight_time_h = (batt_mah / 1000) / I_total
            range_km = V * flight_time_h * 3.6

            results.append({
                "V_ms": V,
                "alpha_deg": alpha_level_flight_deg,
                "Thrust_req_N": thrust_required_N,
                "P_elec_W": P_elec_total,
                "I_total_A": I_total,
                "Flight_Time_min": flight_time_h * 60,
                "Range_km": range_km,
                "g_W": gW,
                "Throttle_pct": thr
            })

        except (ValueError, RuntimeError):
            # brentq fails if L=W is not possible in the given alpha range (e.g., too slow)
            continue
    
    if not results:
        print("\nERROR: Performance sweep failed. The aircraft may be unable to sustain level flight.")
        exit(0)

    perf_df = pd.DataFrame(results)

    # Find optimal points
    idx_endurance = perf_df['P_elec_W'].idxmin()
    idx_range = perf_df['Range_km'].idxmax()

    endurance_pt = perf_df.loc[idx_endurance]
    range_pt = perf_df.loc[idx_range]
    
    # Max level flight speed is the last valid point in our sweep
    V_max_level_flight = perf_df["V_ms"].iloc[-1]


    # ==================================================
    # RESULTS
    # ==================================================
    print("\n" + "="*40)
    print("OVERALL SUMMARY:")
    print("="*40)
    print(f"Total Fuselage Length : {fuselage_length:.4f} m")
    print(f"Total Mass            : {total_mass_kg:.4f} kg ({weight_N:.2f} N)")
    print(f"Center of Gravity (CG_x): {cg_x_m:.4f} m (from fuselage front, x=0)")
    print()
    print(f"--- Motors & Propulsion ---")
    print(f"Motor name             : {motor_model.brand} {motor_model.motor} {motor_model.kv}KV")
    print(f"Motor count            : {num_motors}")
    print(f"Propellers             : {motor_model.prop}")
    print(f"Total Static Thrust    : {static_thrust_N:.2f} N ({static_thrust_g/1000:.2f} kgf)")
    print()

    print(f"--- Aerodynamic Properties ---")
    print(f"Reference Area (S_ref)                   = {S_ref:.4f} m² (Total main wing planform)")
    print(f"Effective Aspect Ratio (AR_eff)          = {AR_eff:.2f}")
    print(f"Calculated Oswald Efficiency (e)         = {e_oswald:.3f}")
    print(f"Induced Drag Factor (k = 1/(π*AR*e))     = {k_induced_drag:.4f}")
    print(f"Total Zero-lift drag coeff (C_D₀)        = {C_D0:.5f}")
    print(f"  (Main Wing: {CD0_contrib_main/C_D0*100:.1f}%, Tail: {CD0_contrib_tail/C_D0*100:.1f}%, Fuse: {CD0_contrib_fuse/C_D0*100:.1f}%, Interf: {C_D0_interf_factor*100:.1f}%)")
    print()
    print(f"--- Stability ---")
    print(f"Neutral Point (x_np from nose)           = {x_np:.3f} m")
    print(f"Static Margin (SM)                       = {static_margin_pct:.1f} %c_ref")
    print(f"Lift curve slope (CLα)                   = {CLalpha_deg:.4f} /deg")
    print(f"Moment curve slope (Cmα about xyz_ref)   = {Cmalpha_deg:.4f} /deg")
    print()

    print(f"--- Performance Estimates (Level Flight) ---")
    print(f"2D Airfoil C_Lmax                        = {CL_max_airfoil_2D:.3f} at alpha={alpha_stall_airfoil_2D:.1f} deg")
    print(f"Estimated 3D Aircraft C_Lmax             = {CL_max_3D:.3f}")
    print(f"Stall Speed (V_stall)                    = {V_stall:.2f} m/s ({V_stall*3.6:.1f} km/h)")
    print(f"Max Level Flight Speed (V_max)           = {V_max_level_flight:.2f} m/s ({V_max_level_flight*3.6:.1f} km/h)")
    print("-" * 50)
    
    print("--- Max Endurance Point ---")
    print(f"  Airspeed                               = {endurance_pt['V_ms']:.2f} m/s ({endurance_pt['V_ms']*3.6:.1f} km/h)")
    print(f"  Angle of Attack                        = {endurance_pt['alpha_deg']:.2f} deg")
    print(f"  Thrust Required                        = {endurance_pt['Thrust_req_N']:.2f} N")
    print(f"  Total Electrical Power                 = {endurance_pt['P_elec_W']:.2f} W")
    print(f"  Total Current Draw                     = {endurance_pt['I_total_A']:.2f} A")
    print(f"  Motor Throttle (estimated)             = {endurance_pt['Throttle_pct']:.1f} %")
    print(f"  Flight Time                            = {endurance_pt['Flight_Time_min']:.1f} min")
    print(f"  Range at this speed                    = {endurance_pt['Range_km']:.1f} km")
    print("-" * 50)

    print("--- Max Range Point ---")
    print(f"  Airspeed                               = {range_pt['V_ms']:.2f} m/s ({range_pt['V_ms']*3.6:.1f} km/h)")
    print(f"  Angle of Attack                        = {range_pt['alpha_deg']:.2f} deg")
    print(f"  Thrust Required                        = {range_pt['Thrust_req_N']:.2f} N")
    print(f"  Total Electrical Power                 = {range_pt['P_elec_W']:.2f} W")
    print(f"  Total Current Draw                     = {range_pt['I_total_A']:.2f} A")
    print(f"  Motor Throttle (estimated)             = {range_pt['Throttle_pct']:.1f} %")
    print(f"  Flight Time at this speed              = {range_pt['Flight_Time_min']:.1f} min")
    print(f"  Max Range                              = {range_pt['Range_km']:.1f} km")
    print("-" * 50)
    
    if arguments.export:
        airplane.export_OpenVSP_vspscript(f"{arguments.export}.vspscript")

    # ==================================================
    # VISUALISATION
    # ==================================================
    
    if arguments.view: 
        airplane.draw_three_view()
    if arguments.vlm:
        # Draw the VLM at the max range flight condition for a more relevant visualization
        vlm_range = asb.VortexLatticeMethod(
            airplane=airplane,
            op_point=asb.OperatingPoint(velocity=range_pt['V_ms'], alpha=range_pt['alpha_deg']),
        )
        vlm_range.draw(show_kwargs=dict())
    
    # Optional: Plot the performance curves
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(perf_df['V_ms'], perf_df['P_elec_W'], 'b-', label='Power Required (W)')
    plt.plot(endurance_pt['V_ms'], endurance_pt['P_elec_W'], 'bo', markersize=10, label=f'Min Power Point ({endurance_pt["V_ms"]:.1f} m/s)')
    plt.title('Performance Curves')
    plt.xlabel('Airspeed (m/s)')
    plt.ylabel('Power (W)', color='b')
    plt.grid(True)
    plt.legend()
    
    ax2 = plt.gca().twinx()
    plt.plot(perf_df['V_ms'], perf_df['Thrust_req_N'], 'r-', label='Thrust Required (N)')
    plt.plot(airspeeds_to_test, get_thrust_available(airspeeds_to_test, V_pitch_effective, static_thrust_N, num_motors), 'g--', label='Thrust Available (N)')
    plt.ylabel('Thrust (N)', color='r')
    plt.legend(loc='lower right')

    plt.subplot(2, 1, 2)
    plt.plot(perf_df['V_ms'], perf_df['Range_km'], 'k-', label='Range (km)')
    plt.plot(range_pt['V_ms'], range_pt['Range_km'], 'ko', markersize=10, label=f'Max Range Point ({range_pt["V_ms"]:.1f} m/s)')
    plt.xlabel('Airspeed (m/s)')
    plt.ylabel('Range (km)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{arguments.assembly}_performance_curves.png')
    #plt.show()