#!/usr/bin/env bash
set -euo pipefail

ASSY="${1:-assembly_quadrocket10}"

SRC="${ASSY}.vspscript"            # AeroSandbox output
AS_VSP="${ASSY}_native.vspscript"  # script we will run
DEG="${ASSY}_DegenGeom.csv"        # degen-geom file
LOG="${ASSY}_vsp.log"

echo "Inject degen call"
perl -0777 -pe "s/}\\s*\\z/    Update();\n    SetComputationFileName(DEGEN_GEOM_CSV_TYPE, \"$DEG\");\n    ComputeDegenGeom(SET_ALL, DEGEN_GEOM_CSV_TYPE);\n}\n/s" \
     "$SRC" > "$AS_VSP"

echo "Run vspscript head-less"
VSP="$(command -v vspscript || command -v vsp)"
: "${VSP:?OpenVSP executable not found}"

"$VSP" -script "$AS_VSP" >"$LOG" 2>&1 || true

if [[ -s "$DEG" ]]; then
    echo "✓  DONE – degen-geom at  $DEG"
else
    echo "✗  OpenVSP failed – tail of log:"
    [[ -s "$LOG" ]] && tail -n 40 "$LOG"
    exit 3
fi
