#!/bin/bash
# Stage a zipped zarr store from hnvme to node-local scratch.
#
# Usage:
#   stage_zarr_zip.sh <source.zarr.zip> <destination.zarr>
#
# The zip should be stored without outer compression (`zip -0`). ripunzip is
# preferred for parallel extraction; unzip is used as a portability fallback.

set -euo pipefail

SRC_ZIP=${1:?"Usage: stage_zarr_zip.sh <source.zarr.zip> <destination.zarr>"}
DST_DIR=${2:?"Usage: stage_zarr_zip.sh <source.zarr.zip> <destination.zarr>"}

NODE_ID=${SLURM_NODEID:-0}
TMP_DIR="${DST_DIR}.tmp.${SLURM_JOB_ID:-manual}.${NODE_ID}"
PARENT_DIR=$(dirname "$DST_DIR")

if [ ! -f "$SRC_ZIP" ]; then
    echo "Node ${NODE_ID}: missing source zip: $SRC_ZIP" >&2
    exit 1
fi

if [ -e "$DST_DIR" ]; then
    if { [ -e "$DST_DIR/.zgroup" ] && [ -e "$DST_DIR/.zmetadata" ]; } || [ -e "$DST_DIR/zarr.json" ]; then
        echo "Node ${NODE_ID}: staged zarr already exists: $DST_DIR"
        exit 0
    fi

    echo "Node ${NODE_ID}: removing incomplete staged zarr: $DST_DIR"
    rm -rf "$DST_DIR"
fi

rm -rf "$TMP_DIR"
mkdir -p "$PARENT_DIR"

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

mkdir -p "$TMP_DIR"
if [ "$(basename "$EXTRACTOR")" = "ripunzip" ]; then
    "$EXTRACTOR" unzip-file -d "$TMP_DIR" "$SRC_ZIP"
else
    unzip -q "$SRC_ZIP" -d "$TMP_DIR"
fi

if [ ! -e "$TMP_DIR/.zgroup" ] && [ ! -e "$TMP_DIR/zarr.json" ]; then
    echo "Node ${NODE_ID}: extracted output does not look like a zarr store: $TMP_DIR" >&2
    rm -rf "$TMP_DIR"
    exit 1
fi

if [ -e "$TMP_DIR/.zgroup" ] && [ ! -e "$TMP_DIR/.zmetadata" ]; then
    echo "Node ${NODE_ID}: extracted v2 zarr is missing consolidated metadata: $TMP_DIR/.zmetadata" >&2
    rm -rf "$TMP_DIR"
    exit 1
fi

mv "$TMP_DIR" "$DST_DIR"

end=$(date +%s.%N)
echo "Node ${NODE_ID}: staged zarr to $DST_DIR in $(awk "BEGIN {print ${end} - ${start}}") seconds"
