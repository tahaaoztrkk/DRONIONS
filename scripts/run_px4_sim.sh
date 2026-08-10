#!/bin/bash
# Terminal 1: PX4 SITL + Gazebo, forced onto the NVIDIA dGPU.
#
# Without the offload vars below, Mesa opens the NVIDIA DRM node on this
# hybrid AMD-iGPU/NVIDIA-dGPU laptop, finds no usable driver there
# ("libEGL warning: ... driver (null)" -> "failed to create dri2 screen")
# and silently falls back to a renderer slow enough that the GUI stalls the
# simulation. PX4 runs in lockstep with Gazebo, so that stall freezes the
# flight stack too and MAVROS drops the link mid-flight.
#
# Measured: 5 EGL warnings and no GPU process by default; 0 warnings and
# "gz sim gui" resident on the RTX with these set.
#
# Usage: scripts/run_px4_sim.sh [make-target]
set -euo pipefail

PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
TARGET="${1:-gz_x500_dronions_dronions_scenario}"

cd "$PX4_DIR"

# PX4's build tooling needs its own venv (kconfiglib etc.), not the DRONIONS one.
if [ -f px4venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source px4venv/bin/activate
fi

export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

# VS Code's snap sets GTK_PATH and friends into /snap/code/..., which breaks
# Gazebo's GUI when launched from an integrated terminal.
exec env -u GTK_PATH -u GTK_EXE_PREFIX -u GDK_PIXBUF_MODULE_FILE \
    make px4_sitl "$TARGET"
