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


import os
import sys
import json
import dgl
import pyvista as pv
import torch
import hydra
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import Rbf, griddata
from hydra.utils import to_absolute_path
from torch.cuda.amp import GradScaler
from omegaconf import DictConfig

from physicsnemo.distributed import DistributedManager
from physicsnemo.models.meshgraphnet import MeshGraphNet
from physicsnemo.datapipes.cae.readers import read_vtp

from preprocessor import fetch_mesh_vertices, convert_to_triangular_mesh

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from dataloader import create_dataloader
from utils import (
    find_bin_files,
    count_trainable_params,
)


def load_model_params(model, filename):
    """Load the model parameters from a checkpoint file."""
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        state_dict = remove_module_prefix(checkpoint["model_state_dict"])
        model.load_state_dict(state_dict)
        print(f"Checkpoint loaded: {filename}")
    else:
        print(f"No checkpoint found at {filename}")


def remove_module_prefix(state_dict):
    """Remove the 'module.' prefix from the state_dict keys."""
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'module.' prefix from the keys
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v
    return new_state_dict


def print_memory_usage(tag=""):
    """Print the memory usage."""
    allocated = torch.cuda.memory_allocated() / (1024**2)  # Convert to MB
    reserved = torch.cuda.memory_reserved() / (1024**2)  # Convert to MB
    print(
        f"{tag} - Allocated Memory: {allocated:.2f} MB, Reserved Memory: {reserved:.2f} MB"
    )


def gather_all_errors(local_errors, world_size):
    """Gather all errors from all processes."""
    # Convert list of errors to tensor
    local_errors_tensor = torch.tensor(local_errors, dtype=torch.float32, device="cuda")

    # Gather errors from all processes
    gathered_errors = [torch.zeros_like(local_errors_tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_errors, local_errors_tensor)

    # Flatten the list of tensors to get all errors in one tensor
    all_errors = torch.cat(gathered_errors)

    return all_errors


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Enable cuDNN auto-tuner
    torch.backends.cudnn.benchmark = cfg.enable_cudnn_benchmark

    # Instantiate the distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    print(f"Rank {dist.rank} of {dist.world_size}")

    # AMP Configs
    amp_dtype = torch.bfloat16
    amp_device = "cuda"

    # Find all .bin files in the directory
    test_dataset = find_bin_files(to_absolute_path(cfg.test_partitions_path))

    # Prepare the stats
    with open(to_absolute_path(cfg.stats_file), "r") as f:
        stats = json.load(f)
    mean = stats["mean"]
    std = stats["std_dev"]

    # Create DataLoader
    test_dataloader = create_dataloader(
        test_dataset,
        mean,
        std,
        batch_size=1,
        prefetch_factor=None,
        use_ddp=True,
        num_workers=4,
        drop_last=False,
    )
    # graphs is a list of graphs, each graph is a list of partitions
    test_graphs = [graph_partitions for graph_partitions, _ in test_dataloader]

    test_ids = [id[0] for _, id in test_dataloader]
    print(f"test dataset size: {len(test_graphs) * dist.world_size}")

    # read the raw .vtp files
    surface_vertices = []
    surface_mesh = []
    for i in test_ids:
        vtp_file = os.path.join(
            to_absolute_path(cfg.data_path), f"run_{i}", f"boundary_{i}.vtp"
        )
        if os.path.exists(vtp_file):
            print(f"Reading {vtp_file}")
            mesh = read_vtp(vtp_file)
            mesh = convert_to_triangular_mesh(mesh)
            mesh = mesh.cell_data_to_point_data()
            vertices = fetch_mesh_vertices(mesh)
            surface_mesh.append(mesh)
            surface_vertices.append(vertices)

    # Initialize model
    model = MeshGraphNet(
        input_dim_nodes=24,
        input_dim_edges=4,
        output_dim=4,
        processor_size=cfg.num_message_passing_layers,
        aggregation="sum",
        hidden_dim_node_encoder=cfg.hidden_dim,
        hidden_dim_edge_encoder=cfg.hidden_dim,
        hidden_dim_node_decoder=cfg.hidden_dim,
        mlp_activation_fn=cfg.activation,
        do_concat_trick=cfg.use_concat_trick,
        num_processor_checkpoint_segments=cfg.checkpoint_segments,
    ).to(device)
    print("Instantiated the model")
    print(f"Number of trainable parameters: {count_trainable_params(model)}")

    # Load the checkpoint
    load_model_params(model, cfg.checkpoint_filename)

    # compile
    # model = torch.jit.script(model)
    # torch._dynamo.reset()
    # model = torch.compile(model, mode="reduce-overhead")

    mean = {key: torch.tensor(value).to(device) for key, value in mean.items()}
    std = {key: torch.tensor(value).to(device) for key, value in std.items()}

    for i in range(len(test_graphs)):
        # Placeholder to accumulate predictions and node features for the full graph's nodes
        num_nodes = sum([subgraph.num_nodes() for subgraph in test_graphs[i]])

        # Initialize accumulators for predictions and node features
        pressure_pred = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)
        shear_stress_pred = torch.zeros(
            (num_nodes, 3), dtype=torch.float32, device=device
        )
        pressure_true = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)
        shear_stress_true = torch.zeros(
            (num_nodes, 3), dtype=torch.float32, device=device
        )
        coordinates = torch.zeros((num_nodes, 3), dtype=torch.float32, device=device)
        normals = torch.zeros((num_nodes, 3), dtype=torch.float32, device=device)
        area = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)

        # Accumulate predictions and node features from all partitions
        pressure_l2_error_list = []
        shear_stress_x_l2_error_list = []
        shear_stress_y_l2_error_list = []
        shear_stress_z_l2_error_list = []
        pressure_l1_error_list = []
        shear_stress_x_l1_error_list = []
        shear_stress_y_l1_error_list = []
        shear_stress_z_l1_error_list = []
        for j in range(cfg.num_partitions):
            part = test_graphs[i][j].to(device)

            # Get node features (coordinates and normals)
            ndata = torch.cat(
                (
                    part.ndata["coordinates"],
                    part.ndata["normals"],
                    torch.sin(2 * np.pi * part.ndata["coordinates"]),
                    torch.cos(2 * np.pi * part.ndata["coordinates"]),
                    torch.sin(4 * np.pi * part.ndata["coordinates"]),
                    torch.cos(4 * np.pi * part.ndata["coordinates"]),
                    torch.sin(8 * np.pi * part.ndata["coordinates"]),
                    torch.cos(8 * np.pi * part.ndata["coordinates"]),
                    # part.ndata["sdf"],
                ),
                dim=1,
            )

            with torch.inference_mode():
                with torch.autocast(amp_device, enabled=True, dtype=amp_dtype):
                    pred = model(ndata, part.edata["x"], part)
                    pred_filtered = pred[part.ndata["inner_node"].bool()]
                    target = torch.cat(
                        (part.ndata["pressure"], part.ndata["shear_stress"]),
                        dim=1,
                    )
                    target_filtered = target[part.ndata["inner_node"].bool()]

                    # Store the predictions based on the original node IDs (using `dgl.NID`)
                    original_nodes = part.ndata[dgl.NID]
                    inner_original_nodes = original_nodes[
                        part.ndata["inner_node"].bool()
                    ]

                    # Accumulate the predictions
                    pressure_pred[inner_original_nodes] = (
                        pred_filtered[:, 0:1].clone().to(torch.float32)
                    )
                    shear_stress_pred[inner_original_nodes] = (
                        pred_filtered[:, 1:].clone().to(torch.float32)
                    )

                    # Accumulate the ground truth
                    pressure_true[inner_original_nodes] = (
                        target_filtered[:, 0:1].clone().to(torch.float32)
                    )
                    shear_stress_true[inner_original_nodes] = (
                        target_filtered[:, 1:].clone().to(torch.float32)
                    )

                    # Accumulate the node features
                    coordinates[original_nodes] = (
                        part.ndata["coordinates"].clone().to(torch.float32)
                    )
                    normals[original_nodes] = (
                        part.ndata["normals"].clone().to(torch.float32)
                    )
                    area[original_nodes] = part.ndata["area"].clone().to(torch.float32)

        # Denormalize predictions and node features using the global stats
        pressure_pred_denorm = (
            pressure_pred * torch.tensor(std["pressure"])
        ) + torch.tensor(mean["pressure"])
        shear_stress_pred_denorm = (
            shear_stress_pred * torch.tensor(std["shear_stress"])
        ) + torch.tensor(mean["shear_stress"])
        coordinates_denorm = (
            coordinates * torch.tensor(std["coordinates"])
        ) + torch.tensor(mean["coordinates"])

        # Interpolate onto the original simulation mesh
        k = 5
        coordinates_denorm_np = coordinates_denorm.cpu().numpy()
        surface_vertices_np = surface_vertices[i]

        # Fit the kNN model
        nbrs_surface = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(
            coordinates_denorm_np
        )

        # Find the k nearest neighbors and their distances
        distances, indices = nbrs_surface.kneighbors(surface_vertices_np)

        if k == 1:
            # Use the nearest neighbor (k=1)
            nearest_indices = indices[:, 0]
            pressure_pred_mesh = pressure_pred_denorm[nearest_indices]
            shear_stress_pred_mesh = shear_stress_pred_denorm[nearest_indices]
        else:
            # Weighted kNN interpolation
            # Avoid division by zero by adding a small epsilon
            epsilon = 1e-8
            weights = 1 / (distances + epsilon)
            weights_sum = np.sum(weights, axis=1, keepdims=True)
            normalized_weights = weights / weights_sum

            # Fetch the predictions of the k nearest neighbors
            pressure_neighbors = pressure_pred_denorm[
                indices
            ]  # Shape: (n_samples, k, 1)
            shear_stress_neighbors = shear_stress_pred_denorm[
                indices
            ]  # Shape: (n_samples, k, 3)

            # Compute the weighted average
            pressure_pred_mesh = np.sum(
                normalized_weights[:, :, np.newaxis] * pressure_neighbors.cpu().numpy(),
                axis=1,
            )
            shear_stress_pred_mesh = np.sum(
                normalized_weights[:, :, np.newaxis]
                * shear_stress_neighbors.cpu().numpy(),
                axis=1,
            )

            # Convert back to torch tensors
            pressure_pred_mesh = torch.from_numpy(pressure_pred_mesh).to(device)
            shear_stress_pred_mesh = torch.from_numpy(shear_stress_pred_mesh).to(device)

        node_attributes = surface_mesh[i].point_data
        pressure_true_mesh = (
            torch.tensor(node_attributes["pMeanTrim"]).unsqueeze(1).to(device)
        )
        shear_stress_true_mesh = torch.tensor(
            node_attributes["wallShearStressMeanTrim"]
        ).to(device)

        avg_rel_l2_err_p = torch.norm(
            pressure_pred_mesh - pressure_true_mesh
        ) / torch.norm(pressure_true_mesh)
        avg_rel_l2_err_wss_x = torch.norm(
            shear_stress_pred_mesh[:, 0] - shear_stress_true_mesh[:, 0]
        ) / torch.norm(shear_stress_true_mesh[:, 0])
        avg_rel_l2_err_wss_y = torch.norm(
            shear_stress_pred_mesh[:, 1] - shear_stress_true_mesh[:, 1]
        ) / torch.norm(shear_stress_true_mesh[:, 1])
        avg_rel_l2_err_wss_z = torch.norm(
            shear_stress_pred_mesh[:, 2] - shear_stress_true_mesh[:, 2]
        ) / torch.norm(shear_stress_true_mesh[:, 2])
        avg_rel_l1_err_p = torch.norm(
            pressure_pred_mesh - pressure_true_mesh, p=1
        ) / torch.norm(pressure_true_mesh, p=1)
        avg_rel_l1_err_wss_x = torch.norm(
            shear_stress_pred_mesh[:, 0] - shear_stress_true_mesh[:, 0], p=1
        ) / torch.norm(shear_stress_true_mesh[:, 0], p=1)
        avg_rel_l1_err_wss_y = torch.norm(
            shear_stress_pred_mesh[:, 1] - shear_stress_true_mesh[:, 1], p=1
        ) / torch.norm(shear_stress_true_mesh[:, 1], p=1)
        avg_rel_l1_err_wss_z = torch.norm(
            shear_stress_pred_mesh[:, 2] - shear_stress_true_mesh[:, 2], p=1
        ) / torch.norm(shear_stress_true_mesh[:, 2], p=1)
        pressure_l2_error_list.append(avg_rel_l2_err_p)
        shear_stress_x_l2_error_list.append(avg_rel_l2_err_wss_x)
        shear_stress_y_l2_error_list.append(avg_rel_l2_err_wss_y)
        shear_stress_z_l2_error_list.append(avg_rel_l2_err_wss_z)
        pressure_l1_error_list.append(avg_rel_l1_err_p)
        shear_stress_x_l1_error_list.append(avg_rel_l1_err_wss_x)
        shear_stress_y_l1_error_list.append(avg_rel_l1_err_wss_y)
        shear_stress_z_l1_error_list.append(avg_rel_l1_err_wss_z)
        print(
            f"Average relative L2 error for pressure for run_{test_ids[i]}: {avg_rel_l2_err_p:.4f}"
        )
        print(
            f"Average relative L2 error for x-wall shear stress for run_{test_ids[i]}: {avg_rel_l2_err_wss_x:.4f}"
        )
        print(
            f"Average relative L2 error for y-wall shear stress for run_{test_ids[i]}: {avg_rel_l2_err_wss_y:.4f}"
        )
        print(
            f"Average relative L2 error for z-wall shear stress for run_{test_ids[i]}: {avg_rel_l2_err_wss_z:.4f}"
        )
        print(
            f"Average relative L1 error for pressure for run_{test_ids[i]}: {avg_rel_l1_err_p:.4f}"
        )
        print(
            f"Average relative L1 error for x-wall shear stress for run_{test_ids[i]}: {avg_rel_l1_err_wss_x:.4f}"
        )
        print(
            f"Average relative L1 error for y-wall shear stress for run_{test_ids[i]}: {avg_rel_l1_err_wss_y:.4f}"
        )
        print(
            f"Average relative L1 error for z-wall shear stress for run_{test_ids[i]}: {avg_rel_l1_err_wss_z:.4f}"
        )

        # Save the full mesh after accumulating all partition predictions
        surface_mesh[i].point_data["pMeanTrimPred"] = pressure_pred_mesh.cpu().numpy()
        surface_mesh[i].point_data["wallShearStressMeanTrimPred"] = (
            shear_stress_pred_mesh.cpu().numpy()
        )
        surface_mesh[i] = surface_mesh[i].extract_surface()
        surface_mesh[i].save(f"inference_mesh_{test_ids[i]}.vtp")

        # # Save the full point cloud after accumulating all partition predictions
        # # Create a PyVista PolyData object for the point cloud
        # point_cloud = pv.PolyData(np.array(surface_vertices[i]))
        # point_cloud["coordinates"] = np.array(surface_vertices[i])
        # point_cloud["pressure_pred"] = pressure_pred_mesh.cpu().numpy()
        # point_cloud["shear_stress_pred"] = shear_stress_pred_mesh.cpu().numpy()
        # point_cloud["pressure_true"] = pressure_true_mesh.cpu().numpy()
        # point_cloud["shear_stress_true"] = shear_stress_true_mesh.cpu().numpy()

        # # Save the point cloud
        # point_cloud.save(f"inference_point_cloud_{test_ids[i]}.vtp")
        # print(f"Saved point cloud for run_{test_ids[i]}")

    # Gather all errors from all processes
    all_l2_errors_pressure = gather_all_errors(pressure_l2_error_list, dist.world_size)
    all_l2_errors_shear_stress_x = gather_all_errors(
        shear_stress_x_l2_error_list, dist.world_size
    )
    all_l2_errors_shear_stress_y = gather_all_errors(
        shear_stress_y_l2_error_list, dist.world_size
    )
    all_l2_errors_shear_stress_z = gather_all_errors(
        shear_stress_z_l2_error_list, dist.world_size
    )
    l2_err_pressure = all_l2_errors_pressure.mean().item()
    l2_err_shear_stress_x = all_l2_errors_shear_stress_x.mean().item()
    l2_err_shear_stress_y = all_l2_errors_shear_stress_y.mean().item()
    l2_err_shear_stress_z = all_l2_errors_shear_stress_z.mean().item()
    all_l1_errors_pressure = gather_all_errors(pressure_l1_error_list, dist.world_size)
    all_l1_errors_shear_stress_x = gather_all_errors(
        shear_stress_x_l1_error_list, dist.world_size
    )
    all_l1_errors_shear_stress_y = gather_all_errors(
        shear_stress_y_l1_error_list, dist.world_size
    )
    all_l1_errors_shear_stress_z = gather_all_errors(
        shear_stress_z_l1_error_list, dist.world_size
    )
    l1_err_pressure = all_l1_errors_pressure.mean().item()
    l1_err_shear_stress_x = all_l1_errors_shear_stress_x.mean().item()
    l1_err_shear_stress_y = all_l1_errors_shear_stress_y.mean().item()
    l1_err_shear_stress_z = all_l1_errors_shear_stress_z.mean().item()

    if dist.rank == 0:
        print(f"Average relative L2 error for pressure: {l2_err_pressure:.4f}")
        print(
            f"Average relative L2 error for x-wall shear stress: {l2_err_shear_stress_x:.4f}"
        )
        print(
            f"Average relative L2 error for y-wall shear stress: {l2_err_shear_stress_y:.4f}"
        )
        print(
            f"Average relative L2 error for z-wall shear stress: {l2_err_shear_stress_z:.4f}"
        )
        print(f"Average relative L1 error for pressure: {l1_err_pressure:.4f}")
        print(
            f"Average relative L1 error for x-wall shear stress: {l1_err_shear_stress_x:.4f}"
        )
        print(
            f"Average relative L1 error for y-wall shear stress: {l1_err_shear_stress_y:.4f}"
        )
        print(
            f"Average relative L1 error for z-wall shear stress: {l1_err_shear_stress_z:.4f}"
        )
        with open(to_absolute_path("average_relative_error.json"), "w") as f:
            json.dump(
                {
                    "l2_error_pressure": l2_err_pressure,
                    "l2_error_wall_shear_stress_x": l2_err_shear_stress_x,
                    "l2_error_wall_shear_stress_y": l2_err_shear_stress_y,
                    "l2_error_wall_shear_stress_z": l2_err_shear_stress_z,
                    "l1_error_pressure": l1_err_pressure,
                    "l1_error_wall_shear_stress_x": l1_err_shear_stress_x,
                    "l1_error_wall_shear_stress_y": l1_err_shear_stress_y,
                    "l1_error_wall_shear_stress_z": l1_err_shear_stress_z,
                },
                f,
            )
    print("Inference complete")


if __name__ == "__main__":
    main()
