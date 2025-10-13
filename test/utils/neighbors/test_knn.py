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

import pytest
import torch

from physicsnemo.utils.neighbors import knn
from physicsnemo.utils.neighbors.knn._cuml_impl import knn_impl as knn_cuml
from physicsnemo.utils.neighbors.knn._scipy_impl import knn_impl as knn_scipy


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("k", [1, 5])
@pytest.mark.parametrize("backend", ["cuml", "torch", "scipy", "auto"])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16]
)
def test_knn(device: str, k: int, backend: str, dtype: torch.dtype):
    """
    Basic test for KNN functionality.
    We use a predictable grid of points to ensure the results are valid.
    """
    # Skip cuml tests on CPU as it's not supported
    if backend == "cuml" and device == "cpu":
        pytest.skip("cuml backend not supported on CPU")

    if backend == "scipy" and device == "cuda":
        pytest.skip("scipy backend not supported on CUDA")

    # Generate a grid of query points
    points = torch.linspace(0, 10, 11, device=device)
    x, y, z = torch.meshgrid(points, points, points, indexing="ij")
    query_space_points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)

    # Generate search space points - add small offsets to query points
    offsets = torch.tensor(
        [
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.3, 0.0],
            [0.0, 0.4, 0.0],
            [0.0, 0.0, 0.5],
        ],
        device=device,
    )
    search_space_points = query_space_points[None, :, :] + offsets[:, None, :]
    search_space_points = search_space_points.reshape(-1, 3)

    # Convert to dtype
    search_space_points = search_space_points.to(dtype)
    query_space_points = query_space_points.to(dtype)

    # Run KNN search
    indices, distances = knn(
        search_space_points,
        query_space_points,
        k=k,
        backend=backend,
    )

    # Basic shape checks
    assert indices.shape[0] == query_space_points.shape[0]
    assert indices.shape[1] == k
    assert distances.shape == indices.shape

    # Check that found points are valid indices
    assert (indices >= 0).all()
    assert (indices < search_space_points.shape[0]).all()

    # Check that distances are non-negative and sorted
    assert (distances >= 0).all()
    assert torch.all(
        distances[:, 1:] >= distances[:, :-1]
    )  # Check distances are sorted

    # For k=1, the closest point should be the offset point
    if k <= len(offsets):
        assert (distances <= 0.5).all()  # Max offset is 0.5


@pytest.mark.parametrize("device", ["cuda"])
def test_knn_torch_compile_no_graph_break(device):
    # Only test if torch.compile is available (PyTorch 2.0+)
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available in this version of PyTorch")

    # Prepare test data
    points = torch.randn(207, 3, device=device)
    queries = torch.randn(13, 3, device=device)
    k = 5

    def search_fn(points, queries):
        return knn(
            points,
            queries,
            k=k,
            backend="auto",
        )

    # Run both and compare outputs
    out_eager = search_fn(points, queries)
    compiled_fn = torch.compile(search_fn, fullgraph=True)
    out_compiled = compiled_fn(points, queries)

    # Compare outputs (tuple of tensors)
    for eager, compiled in zip(out_eager, out_compiled):
        assert torch.allclose(eager, compiled, atol=1e-6)


@pytest.mark.parametrize(
    "device",
    [
        "cuda",
        "cpu",
    ],
)
def test_opcheck(device):
    points = torch.randn(100, 3, device=device)
    queries = torch.randn(10, 3, device=device)
    k = 5

    if device == "cuda":
        op = knn_cuml
    else:
        op = knn_scipy

    torch.library.opcheck(op, args=(points, queries, k))


@pytest.mark.parametrize("device", ["cuda", "cpu"])  # cuml only works on CUDA
def test_knn_comparison(device):
    points = torch.randn(53, 3, device=device)
    queries = torch.randn(21, 3, device=device)
    k = 5

    if device == "cuda":
        indices_cuml, distances_A = knn(points, queries, k, backend="cuml")
        indices_torch, distances_B = knn(points, queries, k, backend="torch")
    else:
        indices_scipy, distances_A = knn(points, queries, k, backend="scipy")
        indices_torch, distances_B = knn(points, queries, k, backend="torch")

    # The points may come in different order between implementations if distances are equal
    # So we check that the sum of distances is approximately equal
    assert torch.allclose(distances_A.sum(), distances_B.sum(), atol=1e-5)

    # For each query point, verify both backends found points at similar distances
    # Sort the distances for each query point to compare
    sorted_dist_cuml = torch.sort(distances_A, dim=1)[0]
    sorted_dist_torch = torch.sort(distances_B, dim=1)[0]
    assert torch.allclose(sorted_dist_cuml, sorted_dist_torch, atol=1e-5)


if __name__ == "__main__":
    test_knn(device="cuda", k=5, backend="cuml", dtype=torch.bfloat16)
