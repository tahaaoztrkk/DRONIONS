#!/bin/bash
# Flies the mission with Gazebo's true pose recorded alongside it.
#
# Every position in the run log comes from the estimator that is itself under
# suspicion: in four runs of ten it diverged, reporting the drone tens of
# metres away and then drifting on plausibly from there. Those logs cannot say
# whether PX4 flew the vehicle somewhere or the estimate came loose, because
# both look identical from inside.
#
# Gazebo knows where the drone actually is. Recording that in parallel is the
# only thing that separates the two.
#
#   scripts/run_with_ground_truth.sh        # 3 runs
#   scripts/run_with_ground_truth.sh 5
set -o pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${1:-3}"
TIMEOUT="${TIMEOUT:-420}"

cleanup() {
    pkill -9 -f "gz topic -e" 2>/dev/null
    pkill -9 -f "log_ground_truth" 2>/dev/null
    ps aux | grep -E "[g]z sim|[b]in/px4|[m]avros_node|[p]arameter_bridge|[d]ronions_ros_node_px4" \
        | awk '{print $2}' | xargs -r kill -9 2>/dev/null
}
trap cleanup EXIT INT TERM

for i in $(seq 1 "$RUNS"); do
    cleanup
    sleep 3
    GT="$REPO/logs/gt_run${i}.csv"
    echo "[$i/$RUNS] ucus basliyor $(date +%T)"

    HEADLESS=1 DRONIONS_TARGET=box COMMAND_DELAY=45 DRONIONS_FAKE_VLM=1 \
        timeout "$TIMEOUT" bash "$REPO/scripts/run_sim_chain.sh" \
        >/dev/null 2>&1 &
    CHAIN=$!

    # The logger cannot subscribe before the world exists, and starting it too
    # early just exits with "topic not found".
    for _ in $(seq 1 120); do
        gz topic -l 2>/dev/null | grep -q '/pose/info' && break
        sleep 1
    done
    if ! gz topic -l 2>/dev/null | grep -q '/pose/info'; then
        echo "[$i/$RUNS] pose/info gelmedi, bu kosu atlaniyor"
        kill $CHAIN 2>/dev/null; continue
    fi

    "$REPO/venv/bin/python3" "$REPO/scripts/log_ground_truth.py" --out "$GT" \
        >/dev/null 2>&1 &
    GTPID=$!
    echo "[$i/$RUNS] yer gercegi kaydediliyor -> $(basename "$GT")"

    wait $CHAIN 2>/dev/null
    kill $GTPID 2>/dev/null
    sleep 2
    echo "[$i/$RUNS] bitti, $(wc -l < "$GT" 2>/dev/null || echo 0) satir"
done

echo
echo "Karsilastirma:"
for i in $(seq 1 "$RUNS"); do
    GT="$REPO/logs/gt_run${i}.csv"
    [ -s "$GT" ] || continue
    echo "--- kosu $i ---"
    "$REPO/venv/bin/python3" "$REPO/scripts/log_ground_truth.py" --compare --out "$GT" \
        2>&1 | tail -20
done
