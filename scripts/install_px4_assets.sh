#!/bin/bash
# Installs this repository's PX4 simulation assets into a PX4-Autopilot checkout.
#
# Why these live here rather than there: PX4 looks for models, worlds and
# airframes inside its own tree, so that is where they were originally written
# -- and nothing in either repository was tracking them. `Tools/simulation/gz`
# is a submodule, so they sat as untracked content inside it; a fresh clone, or
# a `git clean` in PX4, would have taken the entire simulation with it. Five
# campaigns of measurements rested on files that existed on exactly one disk.
#
# The copies under px4/ are now the source of truth. This script puts them
# where PX4 can find them.
#
#   scripts/install_px4_assets.sh                  # into ~/PX4-Autopilot
#   PX4_DIR=/path/to/PX4-Autopilot scripts/install_px4_assets.sh
#   scripts/install_px4_assets.sh --check          # report drift, change nothing
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

[ -d "$PX4_DIR" ] || { echo "PX4 checkout bulunamadi: $PX4_DIR"; exit 1; }

GZ="$PX4_DIR/Tools/simulation/gz"
AIRFRAMES="$PX4_DIR/ROMFS/px4fmu_common/init.d-posix/airframes"
CMAKE="$AIRFRAMES/CMakeLists.txt"
AIRFRAME_ID="4022_gz_x500_dronions"

pairs=(
    "$REPO/px4/models/x500_dronions/model.sdf|$GZ/models/x500_dronions/model.sdf"
    "$REPO/px4/models/x500_dronions/model.config|$GZ/models/x500_dronions/model.config"
    "$REPO/px4/models/dronions_cam/model.sdf|$GZ/models/dronions_cam/model.sdf"
    "$REPO/px4/models/dronions_cam/model.config|$GZ/models/dronions_cam/model.config"
    "$REPO/px4/worlds/dronions_scenario.sdf|$GZ/worlds/dronions_scenario.sdf"
    "$REPO/px4/airframes/$AIRFRAME_ID|$AIRFRAMES/$AIRFRAME_ID"
)

drift=0
for pair in "${pairs[@]}"; do
    src="${pair%%|*}"; dst="${pair##*|}"
    if [ ! -f "$dst" ]; then
        echo "  EKSIK    $(basename "$dst")"
        drift=1
    elif ! cmp -s "$src" "$dst"; then
        echo "  FARKLI   $(basename "$dst")"
        drift=1
    else
        echo "  ayni     $(basename "$dst")"
    fi
done

if grep -q "$AIRFRAME_ID" "$CMAKE" 2>/dev/null; then
    echo "  ayni     CMakeLists.txt ($AIRFRAME_ID kayitli)"
else
    echo "  EKSIK    CMakeLists.txt icinde $AIRFRAME_ID kaydi"
    drift=1
fi

if [ "$CHECK" = "1" ]; then
    [ "$drift" = "0" ] && echo "PX4 tarafi guncel." || echo "PX4 tarafi guncel DEGIL."
    exit "$drift"
fi

echo
for pair in "${pairs[@]}"; do
    src="${pair%%|*}"; dst="${pair##*|}"
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
done

# The airframe has to be listed in px4_add_romfs_files or PX4 will not build it
# into the ROMFS, and `make px4_sitl gz_x500_dronions_*` fails with a target it
# cannot find. Inserted after the last 40xx entry rather than appended, to keep
# the list ordered the way PX4 keeps it.
if ! grep -q "$AIRFRAME_ID" "$CMAKE"; then
    python3 - "$CMAKE" "$AIRFRAME_ID" <<'PY'
import re, sys
path, entry = sys.argv[1], sys.argv[2]
text = open(path).read()
hits = list(re.finditer(r'^\t40\d\d_[^\n]*$', text, re.M))
if not hits:
    sys.exit("CMakeLists.txt icinde 40xx girdisi bulunamadi -- elle ekleyin.")
last = hits[-1]
text = text[:last.end()] + f"\n\t{entry}" + text[last.end():]
open(path, 'w').write(text)
print(f"  CMakeLists.txt guncellendi ({entry} eklendi)")
PY
fi

echo
echo "Kuruldu. PX4 ROMFS degistigi icin yeniden derleme gerekir:"
echo "  cd $PX4_DIR && make px4_sitl gz_x500_dronions_dronions_scenario"
