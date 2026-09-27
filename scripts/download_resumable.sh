#!/usr/bin/env bash
# Resumable download that survives connection resets: resumes from the current file size until complete.
# usage: scripts/download_resumable.sh URL OUTPUT
set -u
url="$1"; out="$2"
total=$(curl -sIL "$url" | grep -i '^content-length' | tail -1 | tr -dc '0-9')
for attempt in $(seq 1 200); do
  have=$(stat -c %s "$out" 2>/dev/null || echo 0)
  if [ -n "$total" ] && [ "$have" -ge "$total" ]; then echo "complete: $out ($have bytes)"; exit 0; fi
  echo "attempt $attempt: have $have / $total"
  curl -sSL --connect-timeout 20 --speed-limit 10000 --speed-time 60 -C - -o "$out" "$url" || sleep 5
done
echo "gave up"; exit 1
