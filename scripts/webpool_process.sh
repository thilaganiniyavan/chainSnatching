#!/usr/bin/env bash
# Training-pool processing loop: repeatedly de-duplicate and pre-screen whatever the segment workers have finished,
# so review kits can be made before the whole backlog is cut. Both steps skip work already done. Ends one round
# after the segment workers have stopped.
#   nohup scripts/webpool_process.sh > ~/snatch_web_dataset/meta/log_webpool_process.txt 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
while true; do
    seg_running=0
    pgrep -f "webdata.segment --kind snatch" >/dev/null && seg_running=1
    echo "=== round $(date '+%F %T') (segment workers running: $seg_running)"
    for try in 1 2 3; do   # dedup can catch segments_snatch.json mid-write; retry
        nice -n 5 $PY -m snatchguard.webdata.dedup --kind snatch | grep -E "summary|reference|Error|Traceback" && break
        sleep 30
    done
    nice -n 5 $PY -m snatchguard.webdata.review prescreen --kind snatch | tail -n 1
    $PY -c "from snatchguard.webdata import trainpool as t; print('pool to review:', len(t.pool()))"
    [ "$seg_running" = 0 ] && { echo "=== done $(date '+%F %T')"; break; }
    sleep 600
done
