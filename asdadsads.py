#!/usr/bin/env python
"""
quick 3-D preview of SURFACE_NODE points
"""
from pathlib import Path
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (needed for 3-D)

csv = Path("assembly_quadrocket10_DegenGeom.csv").read_text().splitlines()

# ------- pluck the SURFACE_NODE rows (same trick as before) ----------
start = next(i for i, L in enumerate(csv) if L.startswith("SURFACE_NODE")) + 2
rows  = []
for L in csv[start:]:
    if not L or L[0] in "#0123456789":       # accept data or blank/comment
        if L and L[0].isdigit():
            rows.append(L.strip())
    else:
        break

df = pd.read_csv(
    StringIO("\n".join(rows)),
    header=None, names=["x", "y", "z", "u", "w"]
)

# -------- fast 3-D scatter ------------------------------------------
fig = plt.figure(figsize=(6, 5))
ax  = fig.add_subplot(111, projection="3d")
ax.scatter(df.x, df.y, df.z, s=2, c=df.z, cmap="viridis")

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.set_box_aspect([1,1,0.3])          # nicer proportions
plt.tight_layout()
plt.show()
