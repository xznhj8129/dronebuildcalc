#!/usr/bin/env bash
set -euo pipefail

ASSY="${1:-assembly_quadrocket10}"
DEG_CSV="${ASSY}_DegenGeom.csv"
[[ -s "$DEG_CSV" ]] || { echo "Need $DEG_CSV"; exit 1; }

# ───────── locate binaries ────────────────────────────────────────────────────
VSPAERO="$(command -v vspaero)"                                   || { echo "vspaero not found"; exit 1; }
VSP="$(command -v vsp || command -v vspscript)"                   || { echo "vsp / vspscript not found"; exit 1; }

# ───────── ensure .vsp3 exists ───────────────────────────────────────────────
if [[ ! -f "${ASSY}.vsp3" ]]; then
    [[ -f "${ASSY}.vspscript" ]] || { echo "No geometry source"; exit 1; }
    TMP=$(mktemp --suffix=.vspscript)
    cat "${ASSY}.vspscript" >"$TMP"
    printf '\nUpdate();\nWriteVSPFile("%s.vsp3");\n' "$ASSY" >>"$TMP"
    "$VSP" -script "$TMP"
    rm -f "$TMP"
fi
[[ -f "${ASSY}.vsp3" ]] || { echo "Unable to create ${ASSY}.vsp3"; exit 1; }

# ───────── generate DegenGeom root for vspaero ───────────────────────────────
"$VSP" -degen "${ASSY}.vsp3"
DEG_ROOT="${ASSY}_DegenGeom"
[[ -f "${DEG_ROOT}.csv" ]] || { echo "vsp -degen failed"; exit 1; }

# stub required by solver
: > "${DEG_ROOT}.vspaero"

# ───────── build solver command file ─────────────────────────────────────────
VAP="${ASSY}.vap"
cat >"$VAP"<<EOF
MSET=0 MACH=0.15,0.25,0.05
AOA=-5,15,1
XCG=0
SYM=0
VLMMODE=1
STABILITY
SAVE \$stab\$
EOF

# ───────── run solver, head-less ─────────────────────────────────────────────
"$VSPAERO" -omp 4 "$DEG_ROOT" < "$VAP"

STAB="${ASSY}.stab"
[[ -s "$STAB" ]] || { echo "VSPAERO failed"; exit 2; }
echo "✓  VSPAERO done – $STAB"

# ───────── .stab → .json ─────────────────────────────────────────────────────
POLAR_JSON="${ASSY}_polar.json"
python - <<'PY'
import re, sys, pandas as pd
rows=[]
for ln in open(sys.argv[1]):
    m=re.match(r'\s*ALPHA\s*=\s*([-\d.]+).*CLtot=\s*([-\d.]+).*CDtot=\s*([-\d.]+).*CMtot=\s*([-\d.]+)', ln)
    if m: rows.append([float(x) for x in m.groups()])
pd.DataFrame(rows,columns=['alpha','CL','CD','CM']).to_json(sys.argv[2],
                                                             orient='records',
                                                             float_format='%.5g')
PY "$STAB" "$POLAR_JSON"
echo "✓  Wrote polar → $POLAR_JSON"

# ───────── generate JSBSim XML ───────────────────────────────────────────────
read CGX  <<<"$(awk -F, 'NR==2{print \$4}' "$DEG_CSV")"
read MASS <<<"$(awk -F, 'NR==2{print \$3}' "$DEG_CSV")"

cat >"${ASSY}.xml"<<EOF
<fdm_config name="${ASSY}">
  <mass_balance>
    <total_mass unit="KG">${MASS}</total_mass>
    <xyz unit="M">${CGX} 0 0</xyz>
  </mass_balance>
  <aerodynamics>
EOF
python - <<'PY'
import json, sys
for p in json.load(open(sys.argv[1])):
    print(f'    <polar alpha="{p["alpha"]}"><cl>{p["CL"]}</cl><cd>{p["CD"]}</cd><cm>{p["CM"]}</cm></polar>')
PY "$POLAR_JSON" >>"${ASSY}.xml"
echo "</aerodynamics></fdm_config>" >>"${ASSY}.xml"
echo "✓  JSBSim XML → ${ASSY}.xml"
