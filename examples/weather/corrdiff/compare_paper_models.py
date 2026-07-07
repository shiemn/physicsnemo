#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI wrapper for comparing completed paper-protocol CorrDiff evaluations."""

from __future__ import annotations

from helpers.paper_eval.compare import *  # noqa: F401,F403
from helpers.paper_eval.compare import main


if __name__ == "__main__":
    raise SystemExit(main())
