#!/bin/bash
# Stage a zipped zarr store from hnvme to node-local scratch.
#
# Usage:
#   stage_zarr_zip.sh <source.zarr.zip> <destination.zarr>
# The destination must be a non-symlink .zarr path beneath /scratch/.
#
# The zip should be stored without outer compression (`zip -0`). ripunzip is
# preferred for parallel extraction; unzip is used as a portability fallback.

set -euo pipefail

SRC_ZIP=${1:?"Usage: stage_zarr_zip.sh <source.zarr.zip> <destination.zarr>"}
DST_DIR=${2:?"Usage: stage_zarr_zip.sh <source.zarr.zip> <destination.zarr>"}

NODE_ID=${SLURM_NODEID:-0}

# Cleanup may remove an incomplete store, so reject broad or ambiguous paths
# before creating directories, opening the lock, or deleting anything.
case "$DST_DIR" in
    /scratch/*.zarr) ;;
    *) echo "Refusing unsafe staging destination: $DST_DIR" >&2; exit 2 ;;
esac
case "$DST_DIR/" in
    *"/../"*|*"/./"*|*"//"*|*"/.zarr/"*)
        echo "Refusing ambiguous staging destination: $DST_DIR" >&2
        exit 2
        ;;
esac
CHECK_DIR=$DST_DIR
while [ -n "$CHECK_DIR" ]; do
    if [ -L "$CHECK_DIR" ]; then
        echo "Refusing symlink in staging destination: $CHECK_DIR" >&2
        exit 2
    fi
    CHECK_DIR=${CHECK_DIR%/*}
done
if [ -e "$DST_DIR" ] && [ ! -d "$DST_DIR" ]; then
    echo "Refusing non-directory staging destination: $DST_DIR" >&2
    exit 2
fi

PARENT_DIR=$(dirname "$DST_DIR")
LOCK_FILE="${DST_DIR}.stage.lock"
if [ -L "$LOCK_FILE" ]; then
    echo "Refusing symlink staging lock: $LOCK_FILE" >&2
    exit 2
fi

if [ ! -f "$SRC_ZIP" ]; then
    echo "Node ${NODE_ID}: missing source zip: $SRC_ZIP" >&2
    exit 1
fi

# Multiple one-GPU jobs can share a four-GPU node. Serialize staging per
# destination so colocated jobs reuse one complete extraction instead of
# racing separate multi-hundred-GiB temporary stores into the same path.
mkdir -p "$PARENT_DIR"
exec 9>"$LOCK_FILE"
echo "Node ${NODE_ID}: waiting for staging lock $LOCK_FILE"
flock 9
echo "Node ${NODE_ID}: acquired staging lock $LOCK_FILE"

if [ -e "$DST_DIR" ]; then
    if { [ -e "$DST_DIR/.zgroup" ] && [ -e "$DST_DIR/.zmetadata" ]; } || [ -e "$DST_DIR/zarr.json" ]; then
        echo "Node ${NODE_ID}: staged zarr already exists: $DST_DIR"
        exit 0
    fi

    echo "Node ${NODE_ID}: removing incomplete staged zarr: $DST_DIR"
    rm -rf -- "$DST_DIR"
fi

TMP_DIR=$(mktemp -d "${DST_DIR}.tmp.XXXXXX")
trap 'rm -rf -- "$TMP_DIR"' EXIT

if command -v ripunzip >/dev/null 2>&1; then
    EXTRACTOR=ripunzip
elif [ -x "$HOME/software/bin/ripunzip" ]; then
    EXTRACTOR="$HOME/software/bin/ripunzip"
else
    EXTRACTOR=unzip
fi

echo "Node ${NODE_ID}: staging $(basename "$SRC_ZIP") to $DST_DIR"
echo "Node ${NODE_ID}: source size $(du -h "$SRC_ZIP" | awk '{print $1}')"
echo "Node ${NODE_ID}: extractor $EXTRACTOR"
start=$(date +%s.%N)

if [ "$(basename "$EXTRACTOR")" = "ripunzip" ]; then
    "$EXTRACTOR" unzip-file -d "$TMP_DIR" "$SRC_ZIP"
else
    unzip -q "$SRC_ZIP" -d "$TMP_DIR"
fi

if [ ! -e "$TMP_DIR/.zgroup" ] && [ ! -e "$TMP_DIR/zarr.json" ]; then
    echo "Node ${NODE_ID}: extracted output does not look like a zarr store: $TMP_DIR" >&2
    exit 1
fi

if [ -e "$TMP_DIR/.zgroup" ] && [ ! -e "$TMP_DIR/.zmetadata" ]; then
    echo "Node ${NODE_ID}: extracted v2 zarr is missing consolidated metadata: $TMP_DIR/.zmetadata" >&2
    exit 1
fi

mv "$TMP_DIR" "$DST_DIR"
trap - EXIT

end=$(date +%s.%N)
echo "Node ${NODE_ID}: staged zarr to $DST_DIR in $(awk "BEGIN {print ${end} - ${start}}") seconds"
