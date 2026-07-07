#!/bin/bash -l
#SBATCH -J europa-zarrv2-clean
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --output=output/europa-zarrv2-clean-%j.out
#SBATCH --mail-type=FAILED,TIME_LIMIT,END
#SBATCH --mail-user=simon.hentschel@uni-wuerzburg.de

set -euo pipefail

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80
export APPTAINERENV_http_proxy=${http_proxy}
export APPTAINERENV_https_proxy=${https_proxy}
export APPTAINERENV_HTTP_PROXY=${HTTP_PROXY}
export APPTAINERENV_HTTPS_PROXY=${HTTPS_PROXY}

HNVME=/hnvme/workspace/b214cb11-helma-ecodata/downscaling
SRCZIP=/data/Europe/wuerzburg450_corrdiff.zarr.zip
SRCROOT=wuerzburg450_corrdiff.zarr
TMPDIR_HOST=${HNVME}/Europe/wuerzburg450_corrdiff_v2.zarr.tmpdir
TMPZIP_HOST=${HNVME}/Europe/wuerzburg450_corrdiff_v2.zarr.zip.clean.tmp
DSTZIP_HOST=${HNVME}/Europe/wuerzburg450_corrdiff_v2.zarr.zip
OLDZIP_HOST=${HNVME}/Europe/wuerzburg450_corrdiff_v2.zarr.zip.direct_duplicate_509244
ZARR3_TARGET=/tmp/zarr3_convert_${SLURM_JOB_ID}

if [ -e "$TMPDIR_HOST" ] || [ -e "$TMPZIP_HOST" ]; then
    echo "Refusing to overwrite existing temp output"
    echo "TMPDIR_HOST=$TMPDIR_HOST"
    echo "TMPZIP_HOST=$TMPZIP_HOST"
    exit 1
fi

rm -rf "$ZARR3_TARGET"

apptainer run \
    --bind /tmp:/tmp \
    --bind ${HNVME}:/data/ \
    ${HNVME}/apptainer/corrdiff_ngc.sif \
    python -m pip install --quiet --target "$ZARR3_TARGET" "zarr>=3,<4" "numpy<2"

apptainer run \
    --bind /tmp:/tmp \
    --bind ${HNVME}:/data/ \
    ${HNVME}/apptainer/corrdiff_ngc.sif \
    env PYTHONPATH="$ZARR3_TARGET" python - <<'PY'
import numpy as np
import zarr
from zarr.storage import LocalStore, ZipStore

src_zip = "/data/Europe/wuerzburg450_corrdiff.zarr.zip"
src_root = "wuerzburg450_corrdiff.zarr"
tmp_dir = "/data/Europe/wuerzburg450_corrdiff_v2.zarr.tmpdir"

src_store = ZipStore(src_zip, mode="r")
src = zarr.open_group(store=src_store, path=src_root, mode="r")

dst_store = LocalStore(tmp_dir, read_only=False)
dst = zarr.create_group(store=dst_store, overwrite=True, zarr_format=2)
dst.attrs.update(dict(src.attrs))


def read_array(arr):
    if arr.ndim == 0:
        return np.asarray(arr[()])
    return np.asarray(arr[:])


def read_sample(arr):
    if arr.ndim == 0:
        return read_array(arr)
    if arr.shape[0] <= 512:
        return read_array(arr)
    if arr.shape[0] > 0:
        return np.asarray(arr[:1])
    return read_array(arr)


def concrete_dtype(sample):
    if sample.dtype == object:
        flat = np.asarray(sample).ravel()
        max_len = max([1] + [len(str(value)) for value in flat])
        return np.dtype(f"<U{max_len}")
    return sample.dtype


def coerce_for_dtype(value, dtype):
    value = np.asarray(value)
    if value.dtype == object:
        return value.astype(dtype)
    return value


names = [
    "XLAT",
    "XLAT_U",
    "XLAT_V",
    "XLONG",
    "XLONG_U",
    "XLONG_V",
    "XTIME",
    "cwb",
    "cwb_center",
    "cwb_pressure",
    "cwb_scale",
    "cwb_valid",
    "cwb_variable",
    "era5",
    "era5_center",
    "era5_pressure",
    "era5_scale",
    "era5_valid",
    "era5_variable",
    "time",
]

for name in names:
    arr = src[name]
    sample = read_sample(arr)
    dtype = concrete_dtype(sample)
    print(
        f"copying {name}: shape={arr.shape}, dtype={dtype}, chunks={arr.chunks}",
        flush=True,
    )

    out = dst.create_array(
        name,
        shape=arr.shape,
        chunks=arr.chunks,
        dtype=dtype,
    )
    out.attrs.update(dict(arr.attrs))

    if arr.ndim == 0:
        out[()] = coerce_for_dtype(read_array(arr), dtype)
    elif arr.shape[0] > 512:
        for i in range(arr.shape[0]):
            out[i] = coerce_for_dtype(arr[i], dtype)
            if i % 1000 == 0:
                print(f"  {name}: {i}/{arr.shape[0]}", flush=True)
    else:
        out[:] = coerce_for_dtype(read_array(arr), dtype)

src_store.close()
print("raw v2 directory copy complete", flush=True)
PY

apptainer run \
    --bind ${HNVME}:/data/ \
    ${HNVME}/apptainer/corrdiff_ngc.sif \
    python - <<'PY'
import os
import zipfile

import zarr

tmp_dir = "/data/Europe/wuerzburg450_corrdiff_v2.zarr.tmpdir"
tmp_zip = "/data/Europe/wuerzburg450_corrdiff_v2.zarr.zip.clean.tmp"

zarr.consolidate_metadata(tmp_dir)

g = zarr.open_consolidated(tmp_dir)
print("verified directory arrays:", sorted(g.array_keys()))
print("cwb", g["cwb"].shape, g["cwb"].dtype)
print("era5", g["era5"].shape, g["era5"].dtype)
print("cwb_variable", g["cwb_variable"][:])

seen = set()
with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
    for root, _, files in os.walk(tmp_dir):
        files.sort()
        for filename in files:
            path = os.path.join(root, filename)
            arcname = os.path.relpath(path, tmp_dir)
            if arcname in seen:
                raise RuntimeError(f"duplicate zip entry: {arcname}")
            seen.add(arcname)
            zf.write(path, arcname)

with zipfile.ZipFile(tmp_zip, "r", allowZip64=True) as zf:
    names = zf.namelist()
    if len(names) != len(set(names)):
        raise RuntimeError("zip contains duplicate entries")
    bad = zf.testzip()
    if bad is not None:
        raise RuntimeError(f"zip test failed at {bad}")

store = zarr.ZipStore(tmp_zip, mode="r")
g = zarr.open_consolidated(store)
print("verified zip arrays:", sorted(g.array_keys()))
print("cwb random", g["cwb"][25666].shape, g["cwb"][25666].dtype)
print("era5 random", g["era5"][25666].shape, g["era5"][25666].dtype)
print("cwb_variable zip", g["cwb_variable"][:])
store.close()
PY

if [ -e "$DSTZIP_HOST" ]; then
    if [ -e "$OLDZIP_HOST" ]; then
        echo "Refusing to overwrite preserved direct zip: $OLDZIP_HOST"
        exit 1
    fi
    mv "$DSTZIP_HOST" "$OLDZIP_HOST"
fi

mv "$TMPZIP_HOST" "$DSTZIP_HOST"
rm -rf "$TMPDIR_HOST"
rm -rf "$ZARR3_TARGET"

echo "DONE: $DSTZIP_HOST"
