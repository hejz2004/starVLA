#!/usr/bin/env bash
# Run on ppu-dev. Copy hjz to H20; keep the source and unrelated target files.
# Every directory named playground is always excluded, at any depth.
# Force transfer of every non-excluded file; do not checksum or quick-skip.
set -Eeuo pipefail

if (( $# )); then
    echo "Usage: bash $0 (no arguments; playground exclusion is fixed)" >&2
    exit 2
fi

SRC='/mnt/workspace/Native_Policy/user/hjz'
HOST='root@47.100.136.60'
PORT='1022'
DST='/mnt/nas/lpy/Native_Policy/user/hjz'
KEY='/root/.ssh/nas-rsync-migration'
LOG="$(mktemp /var/tmp/rsync-hjz-to-h20-$(date +%Y%m%d-%H%M%S)-XXXXXX.log)"

test -d "$SRC" || { echo "Source directory not found: $SRC" >&2; exit 1; }
test -r "$KEY" || { echo "SSH key is not readable: $KEY" >&2; exit 1; }
command -v rsync >/dev/null
command -v ssh >/dev/null

SSH=(ssh
    -o IdentitiesOnly=yes
    -o BatchMode=yes
    -o StrictHostKeyChecking=accept-new
    -o ConnectTimeout=30
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=6
    -i "$KEY"
    -p "$PORT"
)
printf -v RSYNC_RSH '%q ' "${SSH[@]}"
printf -v DST_QUOTED '%q' "$DST"

{
    echo "SOURCE=$SRC/"
    echo "DESTINATION=$HOST:$DST/"
    echo "EXCLUDE=playground/ (all depths)"
    echo "FILES=force transfer all non-excluded files (--ignore-times)"
    echo "LOG=$LOG"
} | tee -a "$LOG"

# The destination must have the NAS mounted before creating the hjz directory.
"${SSH[@]}" "$HOST" \
    "command -v rsync >/dev/null && test -d /mnt/nas/lpy && mkdir -p -- ${DST_QUOTED}"

START=$(date +%s)

# Do not add --delete or --delete-excluded: retain unrelated target files,
# including any playground directories that already exist on H20.
set +e
rsync -rlpt \
    --ignore-times \
    --whole-file \
    --exclude='playground' \
    --partial \
    --partial-dir=.rsync-partial \
    --human-readable \
    --info=progress2,stats2 \
    --stats \
    -e "$RSYNC_RSH" \
    "$SRC/" "$HOST:$DST/" 2>&1 | tee -a "$LOG"
PIPE_RC=("${PIPESTATUS[@]}")
set -e

RC=${PIPE_RC[0]}
LOG_RC=${PIPE_RC[1]}
END=$(date +%s)

echo "RSYNC_EXIT=$RC"
echo "LOG_EXIT=$LOG_RC"
echo "ELAPSED_SECONDS=$((END-START))"
echo "LOG=$LOG"

if (( RC != 0 )); then
    echo 'RSYNC_TRANSFER_FAILED'
    exit "$RC"
fi
if (( LOG_RC != 0 )); then
    echo 'RSYNC_TRANSFER_FINISHED_BUT_LOG_FAILED' >&2
    exit "$LOG_RC"
fi
echo 'RSYNC_TRANSFER_OK'
