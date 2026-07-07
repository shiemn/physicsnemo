# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper to find extreme-precipitation candidate days for targeted evaluation.

Scans the reference (truth) precipitation field over the configured dataset and
prints the top days by two paper-relevant criteria:

  * highest precipitation aggregated over the whole domain (domain sum)
  * highest precipitation at any single gridpoint (domain max)

Use the printed timestamps to populate ``conf/base/times/paper/targets8.yaml`` together
with the 3 fixed "historical / mid-future / end-future" dates you choose.

This is a read-only helper — it loads no model and writes nothing.  It reuses the
dataset configuration from ``conf/evaluate.yaml`` (no checkpoints required).

Usage:
    python select_targets.py --config-name=evaluate \\
        dataset.years='[2005]' select.top_n=10 select.stride=1
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from datasets.dataset import init_dataset_from_config, register_dataset


def _precip_channel(dataset, selection) -> int:
    names = [getattr(c, "name", str(c)) for c in dataset.output_channels()]
    if isinstance(selection, int):
        return selection
    for i, name in enumerate(names):
        norm = "".join(ch for ch in str(name).lower() if ch.isalnum())
        if any(tok in norm for tok in ["precip", "rain", "tp", "rr"]):
            return i
    return 0


@hydra.main(version_base="1.2", config_path="conf", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    register_dataset(cfg.dataset.type)
    dataset, _ = init_dataset_from_config(OmegaConf.to_container(cfg.dataset), batch_size=1)

    top_n = int(cfg.get("select", {}).get("top_n", 10)) if cfg.get("select") else 10
    stride = int(cfg.get("select", {}).get("stride", 1)) if cfg.get("select") else 1
    ch = _precip_channel(dataset, cfg.get("select", {}).get("precip_channel", "auto") if cfg.get("select") else "auto")

    times = dataset.time()
    n = len(dataset)
    print(f"Scanning {n} samples (stride={stride}), precip channel index={ch} ...")

    records = []  # (idx, time, domain_sum, domain_max)
    for i in range(0, n, stride):
        image_tar, *_ = dataset[i]
        if isinstance(image_tar, np.ndarray):
            arr = image_tar
        else:
            arr = image_tar.cpu().numpy()
        # denormalize_output expects (B, C, H, W)
        phys = dataset.denormalize_output(arr[np.newaxis])[0]
        field = np.clip(phys[ch], 0.0, None)
        records.append((i, times[i], float(field.sum()), float(field.max())))

    by_sum = sorted(records, key=lambda r: -r[2])[:top_n]
    by_max = sorted(records, key=lambda r: -r[3])[:top_n]

    print("\n=== Top days by DOMAIN-AGGREGATE precipitation ===")
    for idx, t, dsum, dmax in by_sum:
        print(f"  {t}   sum={dsum:.1f}   max={dmax:.2f} mm")

    print("\n=== Top days by SINGLE-GRIDPOINT maximum precipitation ===")
    for idx, t, dsum, dmax in by_max:
        print(f"  {t}   max={dmax:.2f} mm   sum={dsum:.1f}")

    print("\nPaste the chosen timestamps into conf/base/times/paper/targets8.yaml")
    print("(extreme days above + your 3 fixed historical/mid/end dates).")


if __name__ == "__main__":
    main()
