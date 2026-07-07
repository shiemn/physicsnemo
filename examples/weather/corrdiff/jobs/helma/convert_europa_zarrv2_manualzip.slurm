#!/bin/bash -l
#SBATCH -J europa-zarrv2-manual
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --output=output/europa-zarrv2-manual-%j.out
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
TMPZIP_HOST=${HNVME}/Europe/wuerzburg450_corrdiff_v2.zarr.zip.manual.tmp
DSTZIP_HOST=${HNVME}/Europe/wuerzburg450_corrdiff_v2.zarr.zip
OLDZIP_HOST=${HNVME}/Europe/wuerzburg450_corrdiff_v2.zarr.zip.direct_duplicate_509244
ZARR3_TARGET=/tmp/zarr3_convert_${SLURM_JOB_ID}

if [ -e "$TMPZIP_HOST" ]; then
    echo "Refusing to overwrite existing temp zip: $TMPZIP_HOST"
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
import itertools
import json
import math
import os
import zipfile

import numcodecs
import numpy as np
import zarr
from zarr.storage import ZipStore

src_zip = "/data/Europe/wuerzburg450_corrdiff.zarr.zip"
src_root = "wuerzburg450_corrdiff.zarr"
tmp_zip = "/data/Europe/wuerzburg450_corrdiff_v2.zarr.zip.manual.tmp"

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


def jsonable(value):
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return value


def dumps(obj):
    return json.dumps(jsonable(obj), separators=(",", ":"), sort_keys=True).encode()


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
    return value.astype(dtype, copy=False)


def chunk_slices(shape, chunks):
    if len(shape) == 0:
        yield (), ()
        return
    ranges = [range(int(math.ceil(s / c))) for s, c in zip(shape, chunks)]
    for coords in itertools.product(*ranges):
        selection = []
        for coord, size, chunk in zip(coords, shape, chunks):
            start = coord * chunk
            stop = min(start + chunk, size)
            selection.append(slice(start, stop))
        yield coords, tuple(selection)


def chunk_key(array_name, coords):
    suffix = "0" if len(coords) == 0 else ".".join(str(c) for c in coords)
    return f"{array_name}/{suffix}"


src_store = ZipStore(src_zip, mode="r")
src = zarr.open_group(store=src_store, path=src_root, mode="r")
compressor = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)
compressor_config = compressor.get_config()
metadata = {
    ".zgroup": {"zarr_format": 2},
    ".zattrs": dict(src.attrs),
}
entries = set()

with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
    def write_entry(name, data):
        if name in entries:
            raise RuntimeError(f"duplicate zip entry: {name}")
        entries.add(name)
        zf.writestr(name, data)

    write_entry(".zgroup", dumps(metadata[".zgroup"]))
    write_entry(".zattrs", dumps(metadata[".zattrs"]))

    for name in names:
        arr = src[name]
        sample = read_sample(arr)
        dtype = concrete_dtype(sample)
        shape = tuple(int(s) for s in arr.shape)
        chunks = tuple(int(c) for c in arr.chunks)
        print(
            f"copying {name}: shape={shape}, dtype={dtype}, chunks={chunks}",
            flush=True,
        )

        zarray = {
            "zarr_format": 2,
            "shape": list(shape),
            "chunks": list(chunks),
            "dtype": dtype.str,
            "compressor": compressor_config,
            "fill_value": None,
            "order": "C",
            "filters": None,
        }
        zattrs = dict(arr.attrs)
        metadata[f"{name}/.zarray"] = zarray
        metadata[f"{name}/.zattrs"] = zattrs
        write_entry(f"{name}/.zarray", dumps(zarray))
        write_entry(f"{name}/.zattrs", dumps(zattrs))

        for count, (coords, selection) in enumerate(chunk_slices(shape, chunks)):
            if len(shape) == 0:
                data = coerce_for_dtype(read_array(arr), dtype)
            else:
                data = coerce_for_dtype(arr[selection], dtype)
            data = np.ascontiguousarray(data)
            write_entry(chunk_key(name, coords), compressor.encode(data))
            if count % 1000 == 0 and (name in {"cwb", "era5"} or count > 0):
                print(f"  {name}: chunk {count}", flush=True)

    consolidated = {"zarr_consolidated_format": 1, "metadata": dict(metadata)}
    write_entry(".zmetadata", dumps(consolidated))

src_store.close()

with zipfile.ZipFile(tmp_zip, "r", allowZip64=True) as zf:
    names_in_zip = zf.namelist()
    if len(names_in_zip) != len(set(names_in_zip)):
        raise RuntimeError("zip contains duplicate entries")
    bad = zf.testzip()
    if bad is not None:
        raise RuntimeError(f"zip test failed at {bad}")

print(f"manual zip entries: {len(entries)}", flush=True)
PY

apptainer run \
    --bind ${HNVME}:/data/ \
    ${HNVME}/apptainer/corrdiff_ngc.sif \
    python - <<'PY'
import zarr

tmp_zip = "/data/Europe/wuerzburg450_corrdiff_v2.zarr.zip.manual.tmp"

store = zarr.ZipStore(tmp_zip, mode="r")
g = zarr.open_consolidated(store)
print("verified zip arrays:", sorted(g.array_keys()))
print("cwb", g["cwb"].shape, g["cwb"].dtype)
print("era5", g["era5"].shape, g["era5"].dtype)
print("cwb random", g["cwb"][25666].shape, g["cwb"][25666].dtype)
print("era5 random", g["era5"][25666].shape, g["era5"][25666].dtype)
print("cwb_variable", g["cwb_variable"][:])
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
rm -rf "$ZARR3_TARGET"

echo "DONE: $DSTZIP_HOST"
