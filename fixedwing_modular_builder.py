import math
import warnings

import aerosandbox as asb
import aerosandbox.numpy as np
import neuralfoil as nf

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
):
    base_af = asb.Airfoil(airfoil_name)
    dx_sweep = span * math.tan(math.radians(sweep_deg))

    theta = math.radians(ang)
    y_tip = span * math.cos(theta)
    z_tip = span * math.sin(theta)
    # left_side logic is dormant due to symmetric=True in asb.Wing, which handles it.

    af = base_af # Symmetric wing handles airfoil orientation for the mirrored side
    twist = aoa_deg # Symmetric wing handles twist direction relative to flow for mirrored side

    root = asb.WingXSec(
        xyz_le=[x_le, y_le, z_le], # front, horizontal, vertical
        chord=chord_root,
        twist=twist,
        airfoil=af,
        control_surface_is_symmetric=False,
        control_surface_deflection=0.0,
    )
    tip = asb.WingXSec(
        xyz_le=[x_le + dx_sweep, y_tip, z_tip],
        chord=chord_tip,
        twist=twist,
        airfoil=af,
    )
    return asb.Wing(
            name=f"{name_prefix}_{idx+1}",
            symmetric=True, # This creates the other half of the airspeed-wing
            xsecs=[root, tip],
        )

# ==================================================
# BASIC FLIGHT & AIRFRAME PARAMETERS
# ==================================================

modules = {
    "plain-nose": {
        "type": "nose",
        "len": 0.043,    # m
        "diam": 0.062,   # m
        "mass": 0.015,   # kg
        "cg": 0.5, 
    },
    "dome-nose": {
        "type": "nose",
        "len": 0.043,    # m
        "diam": 0.062,   # m
        "mass": 0.017,   # kg
        "cg": 0.5,
        "eo": True,
    },
    "cam-nose": {
        "type": "nose",
        "len": 0.060,
        "diam": 0.062,
        "mass": 0.029,
        "cg": 0.5,
        "eo": True,
    },
    "avionics-1": {
        "type": "body",
        "len": 0.127,
        "diam": 0.062,
        "mass": 0.085,
        "cg": 0.5,
        "avionics": True,
    },
    "batt-1": {
        "type": "body",
        "len": 0.170,
        "diam": 0.062,
        "mass": 0.089,
        "cg": 0.5,
        "battery": True,
    },
    "payload-1": {
        "type": "body",
        "len": 0.122,
        "diam": 0.062,
        "mass": 0.055,
        "cg": 0.5,
        "payload": True
    },
    "servo-canards-1": {
        "type": "wingmount",
        "len": 0.088,
        "diam": 0.062,
        "mass": 0.125,
        "cg": 0.5,
        "wings_n": 4,
        "wings_ang": 45
    },
    "crossfin-holder": {
        "type": "wingmount",
        "len": 0.172,
        "diam": 0.062,
        "mass": 0.104,
        "cg": 0.5,
        "wings_n": 4,
        "wings_ang": 45
    },
    "4wing-holder": {
        "type": "wingmount",
        "len": 0.150,
        "diam": 0.062,
        "mass": 0.104,
        "cg": 0.5,
        "wings_n": 4,
        "wings_ang": 30
    },
    "2wing-holder": {
        "type": "wingmount",
        "len": 0.150,
        "diam": 0.062,
        "mass": 0.096,
        "cg": 0.5,
        "wings_n": 2,
        "wings_ang": 5
    },
    "2wing-holder-tail": {
        "type": "wingmount",
        "len": 0.150,
        "diam": 0.062,
        "mass": 0.096,
        "cg": 0.5,
        "wings_n": 2,
        "wings_ang": 40
    },
    "short-wing": {
        "type": "wing",
        "span": 0.150,
        "mass": 0.059,
        "cg": 0.5,
        "chord_root": 0.110,
        "chord_tip": 0.110,
        "aoa_ang": 0.0,
        "sweep_ang": 0.0,
        "airfoil": "naca6409",
        "motor": False,
        "ctl_surface": False
    },
    "canard-1": {
        "type": "wing",
        "span": 0.047,
        "mass": 0.005,
        "cg": 0.5,
        "chord_root": 0.065,
        "chord_tip": 0.038,
        "aoa_ang": 0.0,
        "sweep_ang": 10.0,
        "airfoil": "naca0008",
        "motor": False,
        "ctl_surface": True,
        "all_moving": True,
    },
    "220mm-wing-fat": {
        "type": "wing",
        "span": 0.220,
        "mass": 0.040,
        "cg": 0.5,
        "chord_root": 0.11,
        "chord_tip": 0.10,
        "aoa_ang": 0.0,
        "sweep_ang": 1.0,
        "airfoil": "naca6409",
        "motor": False,
        "ctl_surface": False
    },
    "220mm-wing": { #test
        "type": "wing",
        "span": 0.220,
        "mass": 0.030,
        "cg": 0.5,
        "chord_root": 0.105,
        "chord_tip": 0.105,
        "aoa_ang": 0.0,
        "sweep_ang": 0.0,
        "airfoil": "naca6409",
        "motor": False,
        "ctl_surface": False
    },
    "test-wing": { #test
        "type": "wing",
        "span": 0.3,
        "mass": 0.040,
        "cg": 0.5,
        "chord_root": 0.105,
        "chord_tip": 0.105,
        "aoa_ang": 0.0,
        "sweep_ang": 0.0,
        "airfoil": "naca6409",
        "motor": False,
        "ctl_surface": False
    },
    "350mm-wing": { #test
        "type": "wing",
        "span": 0.35,
        "mass": 0.050,
        "cg": 0.5,
        "chord_root": 0.11,
        "chord_tip": 0.09,
        "aoa_ang": 0.0,
        "sweep_ang": 2.0,
        "airfoil": "naca6409",
        "motor": False,
        "ctl_surface": False
    },
    "fixedfin": { #test
        "type": "wing",
        "span": 0.1,
        "mass": 0.005,
        "cg": 0.5,
        "chord_root": 0.1,
        "chord_tip": 0.075,
        "aoa_ang": 0.0,
        "sweep_ang": 10.0,
        "airfoil": "naca6409",
        "motor": False,
        "ctl_surface": False,
        "all_moving": False,
    },
    "saggerwing-7in": {
        "type": "wing",
        "span": 0.135,
        "mass": 0.059,
        "cg": 0.5,
        "chord_root": 0.125,
        "chord_tip": 0.097,
        "aoa_ang": 0.0,
        "sweep_ang": 0.0,
        "airfoil": "naca0008",
        "motor": True,
        "motor_n": 1,
        "motor_x_pos": 0,
        "ctl_surface": False
    },
    "joint-ring-c": {
        "type": "coupler",
        "len": 0.0,#zero apparent length, 0.020,
        "diam": 0.062,
        "mass": 0.02136, #0.010 + 16x m3x6
        "cg": 0.5
    },
    # in tail modules, motor
    "endcap": {
        "type": "tail",
        "len": 0.070,
        "diam": 0.062,
        "mass": 0.027,
        "cg": 0.5,
        "motor": False,
    },
    "vtail": {
        "type": "tail",
        "len": 0.220,
        "diam": 0.062,
        "mass": 0.141,
        "cg": 0.5,
        "motor": True,
        "motor_n": 1,
        "motor_x_pos": 1.0,
    },
    "motor_cap": {
        "type": "tail",
        "len": 0.050, #0.09
        "diam": 0.062,
        "mass": 0.05, #0.099,
        "cg": 0.5,
        "motor": True,
        "motor_n": 1,
        "motor_x_pos": 1.0,
    },
}

parts = {
    "motor": {
        "2807-1300": {
            "mass": 0.047,
            "kv": 1300,
            "minprop": 5,
            "maxprop": 8,
            "len": 0.030,
            "diam": 0.034,
        },
        "2217-1250": {
            "mass": 0.071,
            "kv": 1250,
            "minprop": 8,
            "maxprop": 10,
            "len": 0.075,
            "diam": 0.028,
        }
    },
    "battery": {
        "lipo-4S-2200": {
            "type": "lipo",
            "mass": 0.203,
            "s": 4,
            "p": 1,
            "mah": 2200,
            "c": 50
        },
        "lipo-6S-2200": {
            "type": "lipo",
            "mass": 0.302,
            "s": 6,
            "p": 1,
            "mah": 2200,
            "c": 70
        }
    }
}
assembly_quadrocket = [
    {
        "module":"plain-nose"
    },{
        "module": "servo-canards-1",
        "surfaces": "canard-1",
        "control": ["p","y","r"]
    },{ 
        "module":"batt-1",
        "battery":"lipo-6S-2200"
    },{
        "module":"avionics-1",
    },{
        "module":"crossfin-holder",
        "surfaces": "saggerwing-7in",
        "motors": "2807-1300",
        "props": "7x4.5x3"
    },{
        "module":"endcap"
    }
]

assembly_lancet = [
    {
        "module":"plain-nose"
    },{
        "module":"payload-1",
        "payload_mass": 0.250,
        "parts": [],
    },{
        "module":"batt-1",
        "battery":"lipo-4S-2200"
    },{
        "module": "4wing-holder",
        "surfaces": "350mm-wing",
        "control": ["r"]
    },{ 
        "module":"avionics-1",
    },{
        "module":"4wing-holder",
        "surfaces": "short-wing",
        "control": ["y","p"]
    },{
        "module":"motor_cap",
        "motors": "2807-1300",
        "props": "7x4.5x3",
        "parts": [],
    }
]
assembly_quadlancet = [
    {
        "module":"cam-nose"
    },{
        "module":"payload-1",
        "payload_mass": 0.350,
        "parts": [],
    },{
        "module":"batt-1",
        "battery":"lipo-6S-2200"
    },{
        "module": "4wing-holder",
        "surfaces": "test-wing",
        "control": ["r"]
    },{ 
        "module":"avionics-1",
    },{
        "module":"crossfin-holder",
        "surfaces": "saggerwing-7in",
        "motors": "2807-1300",
        "props": "7x4.5x3"
    },{
        "module":"endcap"
    }
]
assembly_rocket = [
    {
        "module":"plain-nose",
        "parts": [],
    },{ 
        "module":"batt-1",
        "battery":"lipo-4S-2200",
        "parts": [],
    },{
        "module":"crossfin-holder",
        "surfaces": "fixedfin",
        "parts": [],
    },{
        "module":"avionics-1",
        "parts": [],
    },{
        "module": "servo-canards-1",
        "surfaces": "canard-1",
        "control": ["p","y","r"],
        "parts": [],
    },{
        "module":"motor_cap",
        "motors": "2807-1300",
        "props": "7x4.5x3",
        "parts": [],
    }
]
assembly_plane = [
    {
        "module":"cam-nose",
        "parts": [],
    },{ 
        "module":"batt-1",
        "battery":"lipo-4S-2200",
        "parts": [],
    },{
        "module":"payload-1",
        "payload_mass": 0.250,
        "parts": [],
    },{
        "module":"2wing-holder",
        "surfaces": "350mm-wing",
        "parts": [],
    },{
        "module":"avionics-1",
        "parts": [],
    },{
        "module": "2wing-holder-tail",
        "surfaces": "short-wing",
        "control": ["p","y","r"],
        "parts": [],
    },{
        "module":"motor_cap",
        "motors": "2807-1300",
        "props": "7x4.5x3",
        "parts": [],
    }
]


# Couple Assembly
assembly = []
for m in assembly_quadlancet:
    assembly.append(m)
    coupling = {"module":"joint-ring-c"}
    assembly.append(coupling)
assembly.pop()

canards = False

# ==================================================
# SIMULATION PARAMETERS
# ==================================================
altitude_m       = 100.0    # [m]
airspeed         = 30.0     # [m/s]
aoa_deg          = 0.0
static_thrust_kg = 2.0
static_thrust_N  = static_thrust_kg * 9.81

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
print("Airframe:")
print("="*120)
header = (f"{'Module/Part Name':<25} | {'Type':<10} | {'Mass (kg)':>9} | {'Len (m)':>7} | "
          f"{'Local CG (m)':>12} | {'Global CG (m)':>13} | {'Mod Front_x (m)':>14}")
print(header)
print("-" * len(header))

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
            motor_part_data = parts["motor"][motor_name]
            num_motors_per_wing = wing_data.get("motor_n", 1)
            total_motors_on_wings = num_wings * num_motors_per_wing
            motor_mass_each = motor_part_data["mass"]
            total_wing_motors_mass = total_motors_on_wings * motor_mass_each
            
            print(f"{'    ∟ Motor: '+motor_name:<25} | {'on wing':<10} | {total_wing_motors_mass:>9.4f} | {'N/A':>7} | "
                  f"{'(at mount CG)':>12} | {attached_part_cg_global_x_m:>13.4f} |")

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
        battery_mass = battery_part_data["mass"]

        print(f"{'  ∟ Battery: '+battery_name:<25} | {'in body':<10} | {battery_mass:>9.4f} | {'N/A':>7} | "
              f"{'(at body CG)':>12} | {attached_part_cg_global_x_m:>13.4f} |")

        total_mass_kg     += battery_mass
        total_moment_kg_m += battery_mass * attached_part_cg_global_x_m

    if module_data["type"] == "tail" and module_data.get("motor", False):
        motor_name = item_in_assembly["motors"]
        motor_part_data = parts["motor"][motor_name]
        num_tail_motors = module_data.get("motor_n", 1)
        motor_mass_each = motor_part_data["mass"]
        total_tail_motors_mass = num_tail_motors * motor_mass_each

        # Tail motors have specific placement via motor_x_pos (ratio of TAIL module length)
        motor_x_pos_ratio_in_tail = module_data.get("motor_x_pos", 0.5) # Default to mid-module if not specified
        motor_cg_offset_from_module_front_m = module_len * motor_x_pos_ratio_in_tail
        tail_motor_cg_global_x_m = current_x_fuselage_m + motor_cg_offset_from_module_front_m
        
        print(f"{'  ∟ Motor: '+motor_name:<25} | {'in tail':<10} | {total_tail_motors_mass:>9.4f} | {'N/A':>7} | "
              f"{motor_cg_offset_from_module_front_m:>12.4f} | {tail_motor_cg_global_x_m:>13.4f} |")

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
# SURFACE CREATION FUNCTION
# ==================================================
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

# ==================================================
# VORTEX-LATTICE ANALYSIS
# ==================================================
vlm = asb.VortexLatticeMethod(
    airplane=airplane,
    op_point=asb.OperatingPoint(velocity=airspeed, alpha=aoa_deg),
    spanwise_resolution=25, chordwise_resolution=8
)
aero = vlm.run()

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
# form_factor_fuse = 1.2 # Simpler assumption

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
C_D0_interf_factor = 0.10 # Example 10% (WE DON'T WANT EXAMPLES! WE WANT DATA! FIX THIS!)
C_D0 *= (1 + C_D0_interf_factor)

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
# This is highly dependent on aspect ratio, sweep, taper, etc.
# For low AR X-wings, this factor might be lower.
CL_max_3D_factor = 0.85 # Educated guess, adjust based on wing geometry and experience
CL_max_eff_3D = CL_max_airfoil_2D * CL_max_3D_factor

# ==================================================
# PERFORMANCE CALCULATIONS
# ==================================================
# Effective Aspect Ratio of the aircraft
AR_eff = (airplane.b_ref**2 / airplane.s_ref) if airplane.s_ref != 0 else float('inf')

e_oswald = 0.75  # Assumed Oswald efficiency factor for X-wing (can range 0.7-0.9)
k_induced_drag = 1 / (math.pi * AR_eff * e_oswald) if AR_eff > 0 and e_oswald > 0 else float('inf')

# Total drag coefficient at VLM operating point
C_D_total_op_point = C_D0 + CDi # CDi is from VLM at the specific CL (aero["CL"])
D_total_op_point   = float(C_D_total_op_point * 0.5 * rho * airspeed ** 2 * S_ref)

# Stall speed
V_stall   = float(math.sqrt(2 * weight_N / (rho * S_ref * CL_max_eff_3D))) if CL_max_eff_3D > 0 and rho > 0 and S_ref > 0 else float('inf')

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
ROC_op_point = float('nan')
climb_angle_op_point_rad = float('nan')
if weight_N > 0:
    excess_thrust_op_point = static_thrust_N - D_total_op_point
    climb_angle_op_point_rad = math.asin(excess_thrust_op_point / weight_N) if abs(excess_thrust_op_point / weight_N) <= 1 else float('nan')
    if not math.isnan(climb_angle_op_point_rad):
        ROC_op_point = airspeed * math.sin(climb_angle_op_point_rad)

# Add these new print statements to the "RESULTS" section,
# within the "--- Performance Estimates ---" block:

# Modify the existing "Est. Max L/D" line and add new ones:


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

# Add a new sub-section for operating point specific performance:
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
airplane.export_OpenVSP_vspscript("my_airframe.vspscript")

# ==================================================
# VISUALISATION
# ==================================================
if __name__ == '__main__':
    airplane.draw_three_view()
    vlm.draw(show_kwargs=dict())
    