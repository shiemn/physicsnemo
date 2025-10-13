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

# ruff: noqa: E402

"""
Unit tests for the MeshGraphKAN model.
"""

import os
import random
import sys

import numpy as np
import pytest
import torch

script_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

import common  # noqa: E402
from pytest_utils import import_or_fail  # noqa: E402

dgl = pytest.importorskip("dgl")


@import_or_fail("dgl")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_meshgraphkan_forward(device, pytestconfig, set_physicsnemo_force_te):
    from physicsnemo.models.meshgraphnet import MeshGraphKAN

    torch.manual_seed(0)
    dgl.seed(0)
    np.random.seed(0)

    model = MeshGraphKAN(4, 3, 2).to(device)

    bsize, n_nodes, n_edges = 2, 20, 12
    graphs = []
    for _ in range(bsize):
        src = torch.tensor([np.random.randint(n_nodes) for _ in range(n_edges)])
        dst = torch.tensor([np.random.randint(n_nodes) for _ in range(n_edges)])
        graphs.append(dgl.graph((src, dst)).to(device))
    graph = dgl.batch(graphs)

    node_f = torch.randn(graph.num_nodes(), 4).to(device)
    edge_f = torch.randn(graph.num_edges(), 3).to(device)

    assert common.validate_forward_accuracy(model, (node_f, edge_f, graph))


@import_or_fail("dgl")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_meshgraphkan_constructor(device, pytestconfig, set_physicsnemo_force_te):
    arg_sets = [
        dict(
            input_dim_nodes=random.randint(2, 8),
            input_dim_edges=random.randint(1, 4),
            output_dim=random.randint(1, 5),
            processor_size=random.randint(2, 10),
            num_layers_node_processor=3,
            num_layers_edge_processor=2,
            hidden_dim_node_encoder=256,
            hidden_dim_edge_encoder=128,
            hidden_dim_node_decoder=128,
            num_harmonics=7,
        ),
        dict(
            input_dim_nodes=random.randint(2, 8),
            input_dim_edges=random.randint(1, 4),
            output_dim=random.randint(1, 5),
            processor_size=1,
            num_layers_node_processor=1,
            num_layers_edge_processor=1,
            hidden_dim_node_encoder=64,
            hidden_dim_edge_encoder=64,
            hidden_dim_node_decoder=64,
            num_harmonics=3,
        ),
    ]

    from physicsnemo.models.meshgraphnet import MeshGraphKAN

    for kw in arg_sets:
        model = MeshGraphKAN(**kw).to(device)
        bsize = random.randint(1, 4)
        n_nodes, n_edges = random.randint(8, 15), random.randint(8, 15)
        graph = dgl.batch(
            [dgl.rand_graph(n_nodes, n_edges).to(device) for _ in range(bsize)]
        )
        node_f = torch.randn(graph.num_nodes(), kw["input_dim_nodes"]).to(device)
        edge_f = torch.randn(graph.num_edges(), kw["input_dim_edges"]).to(device)
        out = model(node_f, edge_f, graph)
        assert out.shape == (graph.num_nodes(), kw["output_dim"])


@import_or_fail("dgl")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_meshgraphkan_optims(device, pytestconfig, set_physicsnemo_force_te):
    from physicsnemo.models.meshgraphnet import MeshGraphKAN

    def make_inputs():
        model = MeshGraphKAN(3, 3, 2).to(device)
        bsize = random.randint(1, 6)
        n_nodes, n_edges = random.randint(10, 20), random.randint(10, 20)
        graph = dgl.batch(
            [dgl.rand_graph(n_nodes, n_edges).to(device) for _ in range(bsize)]
        )
        node_f = torch.randn(graph.num_nodes(), 3).to(device)
        edge_f = torch.randn(graph.num_edges(), 3).to(device)
        return model, [node_f, edge_f, graph]

    m, inp = make_inputs()
    assert common.validate_cuda_graphs(m, (*inp,))
    m, inp = make_inputs()
    assert common.validate_jit(m, (*inp,))
    m, inp = make_inputs()
    assert common.validate_amp(m, (*inp,))
    m, inp = make_inputs()
    assert common.validate_combo_optims(m, (*inp,))


@import_or_fail("dgl")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_meshgraphkan_checkpoint(device, pytestconfig, set_physicsnemo_force_te):
    from physicsnemo.models.meshgraphnet import MeshGraphKAN

    m1 = MeshGraphKAN(4, 3, 4).to(device)
    m2 = MeshGraphKAN(4, 3, 4).to(device)

    graph = dgl.rand_graph(12, 18).to(device)
    node_f = torch.randn(12, 4).to(device)
    edge_f = torch.randn(18, 3).to(device)

    assert common.validate_checkpoint(m1, m2, (node_f, edge_f, graph))


@import_or_fail("dgl")
@common.check_ort_version()
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_meshgraphkan_deploy(device, pytestconfig, set_physicsnemo_force_te):
    from physicsnemo.models.meshgraphnet import MeshGraphKAN

    model = MeshGraphKAN(5, 2, 3).to(device)
    graph = dgl.rand_graph(14, 25).to(device)
    node_f = torch.randn(14, 5).to(device)
    edge_f = torch.randn(25, 2).to(device)
    inputs = (node_f, edge_f, graph)

    assert common.validate_onnx_export(model, inputs)
    assert common.validate_onnx_runtime(model, inputs)
