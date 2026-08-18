#!/bin/bash
# Brings up the whole PX4 simulation stack in one go: SITL + Gazebo, MAVROS,
# the sensor bridge and the DRONIONS node.
#
# Two modes. The default pipes one command in and runs unattended, for
# automated testing. INTERACTIVE=1 leaves the node on your keyboard instead --
# use that for anything involving the dialogue layer, since confirmations and
# follow-up questions are a conversation and cannot be scripted ahead of time.
# Only the node ever wants input; PX4, MAVROS and the bridge do not, which is
# why this is one script plus one prompt rather than four terminals.
#
#   INTERACTIVE=1 HEADLESS=0 scripts/run_sim_chain.sh   # talk to the drone
#   scripts/run_sim_chain.sh                 # headless, target "box"
#   HEADLESS=0 scripts/run_sim_chain.sh      # with the Gazebo GUI
#   DRONIONS_TARGET=cup scripts/run_sim_chain.sh
#   WORLD=dronions_tabletop scripts/run_sim_chain.sh   # the table scene
#   LOGDIR=/tmp/mylogs scripts/run_sim_chain.sh
#
# WORLD drives both halves and they have to agree: PX4 picks the world by make
# target, and the bridge subscribes to gz topics whose paths contain the world
# name. Setting only one of them starts a simulation that publishes nothing the
# node can see -- which looks exactly like a camera failure and wasted a couple
# of runs before it was traced.
#
# Logs land in $LOGDIR as px4.log / mavros.log / bridge.log / node.log.
# Deliberately not 'set -u': /opt/ros/.../setup.bash reads unbound variables,
# and under -u that aborts the script -- which then fires the cleanup trap and
# kills the PX4 build mid-flight.
set -o pipefail

LOGDIR="${LOGDIR:-/tmp/dronions-sim}"
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
DRONIONS_DIR="${DRONIONS_DIR:-$HOME/DRONIONS}"
TARGET="${DRONIONS_TARGET:-box}"
WORLD="${WORLD:-dronions_scenario}"
# Climbing over what blocks you is an outdoor answer. The room has 2.5 m walls
# and the sweep's 5.5 m ceiling flew straight over them, leaving the drone
# outside hovering above a wall it could not descend past. Enclosed scenes get
# a ceiling under the walls; the scenario world keeps the high one, since its
# 3 m test wall is there to be climbed.
case "$WORLD" in
    *room*) export DRONIONS_MAX_SEARCH_ALT="${DRONIONS_MAX_SEARCH_ALT:-2.1}" ;;
esac
MAKE_TARGET="${MAKE_TARGET:-gz_x500_dronions_$WORLD}"
# Seconds to wait after the node starts before typing the target, so the drone
# is airborne and searching first.
COMMAND_DELAY="${COMMAND_DELAY:-50}"
HEADLESS="${HEADLESS:-1}"
INTERACTIVE="${INTERACTIVE:-0}"

mkdir -p "$LOGDIR"
# Keep the previous run instead of deleting it. Behaviour here is intermittent
# -- the same command arrives on one run in four -- so the interesting question
# is almost always "what was different last time", and that was unanswerable
# while every start wiped the evidence. Three runs were lost that way.
if [ -f "$LOGDIR/node.log" ]; then
    PREV="$LOGDIR/$(date -r "$LOGDIR/node.log" +%Y%m%d-%H%M%S)"
    mkdir -p "$PREV"
    for f in px4 mavros bridge node; do
        [ -f "$LOGDIR/$f.log" ] && mv "$LOGDIR/$f.log" "$PREV/$f.log"
    done
    echo "[chain] onceki kosu -> $PREV"
fi
# Old runs are worth keeping, but not forever: px4.log alone reaches tens of MB.
ls -1dt "$LOGDIR"/*/ 2>/dev/null | tail -n +11 | xargs -r rm -rf

cleanup() {
    # gz sim is launched with & inside PX4's own script and survives SIGINT,
    # so it has to be killed explicitly or the next run reuses a stale world.
    ps aux | grep -E "[g]z sim|[b]in/px4|[m]avros_node|[p]arameter_bridge|[d]ronions_ros_node_px4" \
        | awk '{print $2}' | xargs -r kill -9 2>/dev/null
}
trap cleanup EXIT INT TERM
cleanup
sleep 2

cd "$PX4_DIR"
# PX4 builds with its own venv (kconfiglib etc.), not the DRONIONS one.
[ -f px4venv/bin/activate ] && source px4venv/bin/activate

GPU_ENV=(__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
         __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json)
# px4-rc.gzsim tests `[ -z "$HEADLESS" ]`, so it is *emptiness* that starts the
# GUI, not the value. HEADLESS=0 therefore suppresses the GUI exactly like
# HEADLESS=1 does -- and since the caller sets it in our environment, it is
# inherited by the child unless removed. Hence -u rather than HEADLESS=0.
HEADLESS_ENV=(-u HEADLESS)
if [ "$HEADLESS" = "1" ]; then HEADLESS_ENV=(); GPU_ENV+=(HEADLESS=1); fi

# VS Code's snap pollutes GTK_* and breaks Gazebo's GUI from an integrated terminal.
env -u GTK_PATH -u GTK_EXE_PREFIX -u GDK_PIXBUF_MODULE_FILE "${HEADLESS_ENV[@]}" \
    "${GPU_ENV[@]}" \
    script -qfec "make px4_sitl $MAKE_TARGET" "$LOGDIR/px4.log" &

until grep -q "Startup script returned successfully" "$LOGDIR/px4.log" 2>/dev/null; do sleep 0.5; done
echo "[chain] PX4 ready $(date +%T)"

source /opt/ros/jazzy/setup.bash
# Must be PX4's "Onboard" instance; the Normal one on 18570 is where the node
# sends its synthetic GCS heartbeat instead.
script -qfec "ros2 launch mavros px4.launch fcu_url:=udp://:0@127.0.0.1:14580" \
    "$LOGDIR/mavros.log" &

cd "$DRONIONS_DIR"
ros2 launch "$DRONIONS_DIR/ros/launch/dronions_px4_bridge.launch.py" \
    world:="$WORLD" > "$LOGDIR/bridge.log" 2>&1 &

until grep -q "Got HEARTBEAT, connected" "$LOGDIR/mavros.log" 2>/dev/null; do sleep 0.5; done
echo "[chain] MAVROS connected $(date +%T)"

source venv/bin/activate
if [ "$INTERACTIVE" = "1" ]; then
    echo "[chain] node in foreground $(date +%T) -- type commands below."
    echo "[chain]   dunya: $WORLD"
    echo "[chain]   'kutuyu bul' -> Enter onaylar, 'h' iptal eder, 'q' kapatir."
    # No pipe on stdin: the dialogue layer asks for confirmation and expects
    # follow-up questions, so the node needs the real keyboard. 'script' still
    # captures the session to the log while passing input through.
    DISPLAY="${DISPLAY:-:0}" script -qfec "python3 ros/dronions_ros_node_px4.py" "$LOGDIR/node.log"
    cleanup
    exit 0
fi

# The blank second line is the confirmation the dialogue layer now asks for.
# Without it the target is proposed and the run waits forever for an answer
# that a closed pipe can never give.
( sleep "$COMMAND_DELAY"; printf '%s\n\n' "$TARGET" ) | DISPLAY="${DISPLAY:-:0}" \
    script -qfec "python3 ros/dronions_ros_node_px4.py" "$LOGDIR/node.log" &
echo "[chain] node started $(date +%T), target '$TARGET' in ${COMMAND_DELAY}s"
wait
