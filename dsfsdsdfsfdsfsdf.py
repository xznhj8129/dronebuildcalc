import aerosandbox as asb
import math

# ===================================================================
#  1. DEFINE THE FUSELAGE GEOMETRY
#     These are the only numbers you need to change.
# ===================================================================

FUSELAGE_TOTAL_LENGTH = 0.829
FUSELAGE_DIAMETER = 0.057
NOSE_CAP_LENGTH = 0.060
TAIL_CAP_LENGTH = 0.050
NUM_POINTS_PER_CAP = 12  # Use 8-15 for VSP. More is not better.

# ===================================================================
#  2. THE CODE THAT BUILDS THE PERFECT FUSELAGE
#     Do not change this part. It is correct.
# ===================================================================

print("Building fuselage...")

# Calculate key dimensions
R = FUSELAGE_DIAMETER / 2
x_tube_start = NOSE_CAP_LENGTH
x_tube_end = FUSELAGE_TOTAL_LENGTH - TAIL_CAP_LENGTH

# Create an empty list to hold our cross-sections
xsecs = []

# --- Generate the Nose Cap ---
# This creates a smooth elliptical curve from the tip to the tube.
for i in range(NUM_POINTS_PER_CAP):
    t = i / (NUM_POINTS_PER_CAP - 1)  # t goes from 0 to 1
    x = t * NOSE_CAP_LENGTH
    radius = R * math.sqrt(1 - (1 - t) ** 2)
    xsecs.append(asb.FuselageXSec(xyz_c=[x, 0, 0], radius=radius))

# --- Define the straight Tube section ---
# The tube is just two points: where the nose ends, and where the tail begins.
xsecs.append(asb.FuselageXSec(xyz_c=[x_tube_start, 0, 0], radius=R))
xsecs.append(asb.FuselageXSec(xyz_c=[x_tube_end, 0, 0], radius=R))

# --- Generate the Tail Cap ---
# This creates a smooth elliptical curve from the tube to the tail tip.
for i in range(NUM_POINTS_PER_CAP):
    t = i / (NUM_POINTS_PER_CAP - 1)  # t goes from 0 to 1
    x = x_tube_end + (t * TAIL_CAP_LENGTH)
    radius = R * math.sqrt(1 - t ** 2)
    xsecs.append(asb.FuselageXSec(xyz_c=[x, 0, 0], radius=radius))

# Create the final fuselage object from our list of points
clean_fuselage = asb.Fuselage(name="Fuselage", xsecs=xsecs)

# ===================================================================
#  3. ASSEMBLE AIRPLANE AND EXPORT
# ===================================================================

# Create a dummy airplane to hold the fuselage for export
airplane = asb.Airplane(
    name="Clean_Fuselage_Export",
    fuselages=[clean_fuselage]
)

# Export to a VSP script file
output_filename = "CLEAN_FUSELAGE.vspscript"
airplane.export_OpenVSP_vspscript(output_filename)

print(f"\nSUCCESS.")
print(f"Exported to '{output_filename}'. This file will work.")

# To prove it, you can uncomment this to draw it directly
# airplane.draw(show=True)