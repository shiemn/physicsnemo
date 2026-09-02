"""Local launcher/staging regressions; no SLURM, GPU, or real /scratch access."""

import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

import pytest


HELMA = Path(__file__).resolve().parents[1] / "jobs" / "helma"

LAUNCH_HARNESS = r'''
nvidia-smi() {
    [ "${HELMA_TEST_GPU_EXIT:-0}" -eq 0 ] || return "$HELMA_TEST_GPU_EXIT"
    printf '%s\n' 'GPU 0: mock'
}
scontrol() { printf '%s\n' test-node; }
bash() { return "${HELMA_TEST_STAGE_EXIT:-0}"; }
srun() {
    for arg in "$@"; do
        case "$arg" in *.sh) return "${HELMA_TEST_STAGE_EXIT:-0}" ;; esac
    done
    printf '%s\n' "$@" >> "$HELMA_TEST_LAUNCH_LOG"
}
source "$1" "${@:2}"
'''


def launch(tmp_path, name, args, **overrides):
    log = tmp_path / "launch.log"
    env = {
        **os.environ,
        "SLURM_JOB_ID": "677191",
        "SLURM_JOB_NODELIST": "test-node",
        "SLURM_NNODES": "2",
        "CUDA_VISIBLE_DEVICES": "0",
        "HELMA_TEST_LAUNCH_LOG": str(log),
        **overrides,
    }
    result = subprocess.run(
        ["bash", "-c", LAUNCH_HARNESS, "helma-test", str(HELMA / name), *args],
        env=env, capture_output=True, text=True, timeout=10,
    )
    return result, log.read_text().splitlines() if log.exists() else []


@pytest.mark.parametrize("job_id", [1, 10000, 10500, 20000, 677191])
@pytest.mark.parametrize("name,args", [
    ("train.slurm", ["config", "--direct"]),
    ("multi_node.slurm", ["config", "europa"]),
    ("eval.slurm", ["config", "test", "--direct"]),
    ("temporal/climate_eval.slurm", ["config", "test"]),
])
def test_shared_port_calculation(tmp_path, job_id, name, args):
    result, arguments = launch(
        tmp_path, name, [*args, "note=two words"], SLURM_JOB_ID=str(job_id)
    )
    assert result.returncode == 0, result.stderr
    port = 20000 + job_id % 20000
    option = (f"--rdzv_endpoint=test-node:{port}" if name == "multi_node.slurm"
              else f"--master_port={port}")
    assert option in arguments
    assert "note=two words" in arguments


@pytest.mark.parametrize("name,mode", [
    ("train.slurm", "taiwan"), ("train.slurm", "europa"),
    ("train.slurm", "scratch"), ("multi_node.slurm", "taiwan"),
    ("multi_node.slurm", "europa"), ("multi_node.slurm", "copy.sh"),
    ("eval.slurm", "taiwan"), ("eval.slurm", "europa"),
])
def test_staging_failure_prevents_launch(tmp_path, name, mode):
    args = ["config", "test", mode] if name == "eval.slurm" else ["config", mode]
    result, arguments = launch(tmp_path, name, args, HELMA_TEST_STAGE_EXIT="7")
    assert result.returncode == 7, result.stderr
    assert not arguments


def test_training_gpu_detection_failure_prevents_launch(tmp_path):
    result, arguments = launch(
        tmp_path, "train.slurm", ["config", "--direct"], HELMA_TEST_GPU_EXIT="9"
    )
    assert result.returncode == 9
    assert not arguments


@pytest.mark.parametrize("domain,data_path", [
    ("taiwan", "/data/cwa_dataset/cwa_dataset.zarr"),
    ("europa", "/data/Europe/wuerzburg450_corrdiff_v2.zarr"),
])
def test_eval_staged_data_and_user_override_order(tmp_path, domain, data_path):
    result, args = launch(tmp_path, "eval.slurm", [
        "config", "test", domain, "dataset.data_path=/data/custom.zarr",
    ])
    assert result.returncode == 0, result.stderr
    assert "/scratch/:/data/" in args
    assert args.index(f"dataset.data_path={data_path}") < args.index(
        "dataset.data_path=/data/custom.zarr"
    )


@pytest.mark.parametrize("destination", [
    "/", "/scratch", "/scratch/", "/scratch/.zarr", "/scratch/data",
    "/hnvme/store.zarr", "relative.zarr", "/scratch/../outside.zarr",
    "/scratch/./store.zarr", "/scratch//store.zarr",
])
def test_production_stager_rejects_unsafe_paths(destination):
    # Invalid paths must be rejected even before checking this nonexistent source.
    result = subprocess.run(
        ["bash", str(HELMA / "stage_zarr_zip.sh"), "missing.zip", destination],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2
    assert "Refusing" in result.stderr


def executable(path, body):
    path.write_text(f"#!{sys.executable}\n" + body)
    path.chmod(0o755)


@pytest.fixture
def staging(tmp_path):
    root = tmp_path.resolve()
    scratch = root / "scratch"
    scratch.mkdir()
    script = root / "stage.sh"
    # Redirect only the literal allowed root in a test copy. Production has no
    # environment override for the safety boundary, and tests never use /scratch.
    source = (HELMA / "stage_zarr_zip.sh").read_text()
    assert "    /scratch/*.zarr)" in source
    script.write_text(source.replace("    /scratch/*.zarr)", f'    "{scratch}/"*.zarr)'))
    archive = root / "source.zarr.zip"
    with zipfile.ZipFile(archive, "w") as store:
        for name in (".zgroup", ".zmetadata", "payload"):
            store.writestr(name, "{}")
    bin_dir = root / "bin"
    bin_dir.mkdir()
    executable(bin_dir / "flock", '''
import fcntl, os, sys
with open(os.environ["HELMA_TEST_LOCK_LOG"], "a") as log:
    log.write("attempt\\n")
fcntl.flock(int(sys.argv[1]), fcntl.LOCK_EX)
''')
    executable(bin_dir / "ripunzip", '''
import os, pathlib, sys, time, zipfile
with open(os.environ["HELMA_TEST_EXTRACT_LOG"], "a") as log:
    log.write("extract\\n")
release = os.environ.get("HELMA_TEST_RELEASE")
deadline = time.monotonic() + 8
while release and not pathlib.Path(release).exists():
    if time.monotonic() > deadline:
        sys.exit(8)
    time.sleep(0.02)
destination = pathlib.Path(sys.argv[3])
if os.environ.get("HELMA_TEST_EXTRACT_FAIL"):
    (destination / "partial").write_text("partial")
    sys.exit(7)
with zipfile.ZipFile(sys.argv[4]) as store:
    store.extractall(destination)
''')
    # macOS date lacks GNU %N; this test-only shim supplies the elapsed-time clock.
    executable(bin_dir / "date", "import time\nprint(time.time())\n")
    env = {
        **os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HELMA_TEST_LOCK_LOG": str(root / "locks.log"),
        "HELMA_TEST_EXTRACT_LOG": str(root / "extract.log"),
    }
    return SimpleNamespace(
        root=root, scratch=scratch, archive=archive, env=env,
        command=["bash", str(script), str(archive)],
    )


def stage(fixture, destination):
    return subprocess.run(
        [*fixture.command, str(destination)], env=fixture.env,
        capture_output=True, text=True, timeout=10,
    )


@pytest.mark.parametrize("kind", ["destination", "parent", "lock", "file"])
def test_staging_preserves_unsafe_targets(staging, kind):
    protected = staging.root / "protected"
    protected.mkdir()
    sentinel = protected / "keep"
    sentinel.write_text("keep")
    destination = staging.scratch / "store.zarr"
    if kind == "destination":
        destination.symlink_to(protected, target_is_directory=True)
    elif kind == "parent":
        (staging.scratch / "link").symlink_to(protected, target_is_directory=True)
        destination = staging.scratch / "link" / "store.zarr"
    elif kind == "lock":
        Path(f"{destination}.stage.lock").symlink_to(sentinel)
    else:
        destination.write_text("keep")
    result = stage(staging, destination)
    assert result.returncode == 2, result.stderr
    assert sentinel.read_text() == "keep"
    if kind == "file":
        assert destination.read_text() == "keep"
    assert not (staging.root / "extract.log").exists()


@pytest.mark.parametrize("version", [2, 3])
def test_staging_replaces_incomplete_store_and_reuses_complete_store(staging, version):
    if version == 3:
        with zipfile.ZipFile(staging.archive, "w") as store:
            store.writestr("zarr.json", "{}")
            store.writestr("payload", "{}")
    destination = staging.scratch / "data with spaces" / "store.zarr"
    destination.mkdir(parents=True)
    (destination / "old-partial").write_text("partial")
    for _ in range(2):
        result = stage(staging, destination)
        assert result.returncode == 0, result.stderr
    assert (destination / "payload").exists()
    assert not (destination / "old-partial").exists()
    assert (staging.root / "extract.log").read_text().splitlines() == ["extract"]
    assert not list(destination.parent.glob("store.zarr.tmp.*"))


@pytest.mark.parametrize("failure", ["extract", "metadata"])
def test_staging_failure_cleans_own_temporary_directory(staging, failure):
    if failure == "extract":
        staging.env["HELMA_TEST_EXTRACT_FAIL"] = "1"
    else:
        with zipfile.ZipFile(staging.archive, "w") as store:
            store.writestr(".zgroup", "{}")
    destination = staging.scratch / "store.zarr"
    result = stage(staging, destination)
    assert result.returncode != 0
    assert not destination.exists()
    assert not list(staging.scratch.glob("store.zarr.tmp.*"))


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "Timed out waiting for staging process"
        time.sleep(0.02)


def test_concurrent_staging_extracts_only_once(staging):
    destination = staging.scratch / "store.zarr"
    release = staging.root / "release"
    staging.env["HELMA_TEST_RELEASE"] = str(release)
    processes = []
    try:
        for _ in range(2):
            processes.append(subprocess.Popen(
                [*staging.command, str(destination)], env=staging.env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            ))
            if len(processes) == 1:
                wait_for(lambda: (staging.root / "extract.log").exists())
        lock_log = staging.root / "locks.log"
        wait_for(lambda: lock_log.exists() and len(lock_log.read_text().splitlines()) == 2)
        assert (staging.root / "extract.log").read_text().splitlines() == ["extract"]
        assert processes[1].poll() is None
        release.touch()
        for process in processes:
            _, stderr = process.communicate(timeout=10)
            assert process.returncode == 0, stderr
        assert (staging.root / "extract.log").read_text().splitlines() == ["extract"]
        assert (destination / "payload").exists()
    finally:
        release.touch()
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=10)
