"""Every `times=` override a submitter emits must name a real config.

`times` is a Hydra config group rooted at conf/base/times/, and only three
files sit at the group root -- everything else lives in a subdirectory. A
submitter that emits a bare name resolves to nothing and the job dies at
Hydra composition, which is only visible in the SLURM log.

These scripts use bash 4 associative arrays and cannot be executed on macOS
(bash 3.2), so the override is reconstructed from the source instead.
"""

import re
from pathlib import Path

import pytest

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
TIMES_GROUP = EXAMPLE_ROOT / "conf" / "base" / "times"
SUBMITTERS = sorted((EXAMPLE_ROOT / "jobs" / "helma").rglob("submit_*.sh"))

# "times=future_confidence/${PERIOD_TIMES[$period]}" -> prefix, array name
TIMES_OVERRIDE = re.compile(r'"times=([A-Za-z0-9_/]*)\$\{(\w+)(?:\[[^\]]*\])?\}([^"]*)"')


def _array_values(source: str, name: str) -> list[str]:
    """Collect the values a bash array or scalar assignment can take."""
    values = []
    block = re.search(rf"declare -A {name}=\((.*?)\n\)", source, re.S)
    if block:
        values += re.findall(r"\]=\"([^\"]+)\"", block.group(1))
    # Plain scalar assignments, possibly built from another array. Delimit the
    # referenced name on both sides -- bash names contain underscores, so a
    # bare \w+ would swallow the literal suffix that follows it.
    for rhs in re.findall(rf'^\s*{name}="([^"]+)"', source, re.M):
        expanded = re.sub(
            r"\$\{(\w+)(?:\[[^\]]*\])?\}", lambda m: f"\x00{m.group(1)}\x01", rhs
        )
        match = re.search(r"\x00(\w+)\x01", expanded)
        if match:
            head = expanded[: match.start()]
            tail = expanded[match.end() :]
            for base in _array_values(source, match.group(1)):
                values.append(head + base + tail)
        else:
            values.append(expanded)
    # Restore any still-unresolved reference (e.g. ${seed}) to its shell form.
    return [re.sub(r"\x00(\w+)\x01", r"${\1}", v) for v in values]


def _substitute_seeds(value: str, source: str) -> list[str]:
    """Expand a literal ${seed} using the SEEDS default in the script."""
    if "${seed}" not in value:
        return [value]
    seeds = re.search(r'SEEDS="\$\{SEEDS:-([^}"]+)\}"', source)
    return [value.replace("${seed}", s) for s in (seeds.group(1).split() if seeds else ["43"])]


@pytest.mark.parametrize("script", SUBMITTERS, ids=lambda p: p.name)
def test_times_overrides_resolve_to_real_configs(script):
    source = script.read_text()
    overrides = TIMES_OVERRIDE.findall(source)
    if not overrides:
        pytest.skip(f"{script.name} emits no times= override")

    checked = 0
    for prefix, array_name, suffix in overrides:
        candidates = _array_values(source, array_name)
        assert candidates, f"could not resolve ${{{array_name}}} in {script.name}"
        for candidate in candidates:
            for value in _substitute_seeds(candidate + suffix, source):
                target = TIMES_GROUP / f"{prefix}{value}.yaml"
                assert target.is_file(), (
                    f"{script.name} emits times={prefix}{value}, but "
                    f"{target.relative_to(EXAMPLE_ROOT)} does not exist. "
                    "Overrides must include the config-group subdirectory."
                )
                checked += 1
    assert checked > 0
