"""The committed future-confidence time configs must match their generator.

conf/base/times/future_confidence/ holds 30 files / ~15.5k lines of literal
timestamps produced by scripts/generate_future_time_configs.py. Both are
committed, so they can drift apart -- a hand-edit to a YAML, or a parameter
change in the generator without a regenerate. This test pins them together.
"""

import filecmp
import importlib.util
import tempfile
from pathlib import Path

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = EXAMPLE_ROOT / "scripts" / "generate_future_time_configs.py"
COMMITTED = EXAMPLE_ROOT / "conf" / "base" / "times" / "future_confidence"


def _load_generator():
    spec = importlib.util.spec_from_file_location("_future_time_gen", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _regenerate(into: Path, gen) -> None:
    for filename, label, years in gen.CONFIGS:
        gen.write_config(into / filename, label, years, seed=42)
        stem = filename.removesuffix("_24h_compatible.yaml")
        for seed in gen.EXTRA_SEEDS:
            gen.write_config(
                into / f"{stem}_seed{seed}_24h_compatible.yaml", label, years, seed=seed
            )


def test_committed_configs_match_generator_output():
    gen = _load_generator()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        _regenerate(out, gen)

        committed = sorted(p.name for p in COMMITTED.glob("*.yaml"))
        regenerated = sorted(p.name for p in out.glob("*.yaml"))
        assert committed == regenerated

        drifted = [
            name
            for name in committed
            if not filecmp.cmp(COMMITTED / name, out / name, shallow=False)
        ]
        assert drifted == [], (
            f"{len(drifted)} config(s) differ from generator output: {drifted[:5]}. "
            "Re-run scripts/generate_future_time_configs.py and commit the result."
        )
