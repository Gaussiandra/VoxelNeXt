import torch
import numpy as np
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

class StreamingDetector:
    def __init__(self, cfg, ckpt_path):
        self.logger = common_utils.create_logger(log_file="./temp_log.log")

        self.dataset = NuScenesDataset(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            logger=self.logger,
            training=False
        )

        self.model = build_network(
            model_cfg=cfg.MODEL, 
            num_class=len(cfg.CLASS_NAMES), 
            dataset=self.dataset
        )

        self.model.load_params_from_file(
            filename=ckpt_path, 
            logger=self.logger, 
            to_cpu=False,
        )
        self.model.cuda()
        self.model.eval()

    def predict_from_normalized_data(self, points):
        data_dict = {
            'points': points,
            'batch_size': 1
        }

        # print("points before prepare_data", data_dict["points"])
        data_dict = self.dataset.prepare_data(data_dict=data_dict)
        data_dict = self.dataset.collate_batch([data_dict])

        # print(data_dict.keys())
        # print(data_dict["points"].shape)
        # print(data_dict["points"][0])
        # print(data_dict["points"][:, 5].sum()) ##????
        # print("use_lead_xyz", data_dict["use_lead_xyz"])
        # print("voxels", data_dict["voxels"].shape)
        # print("voxels[10000]", data_dict["voxels"][10000])
        # print("voxel_coords", data_dict["voxel_coords"].shape)
        # print("voxel_coords[0]", data_dict["voxel_coords"][0])
        # print("voxel_num_points", data_dict["voxel_num_points"].shape)
        # print("batch_size", data_dict["batch_size"])

        # import matplotlib.pyplot as plt

        # suffix = "our"
        # # voxelized pointcloud
        # fig, ax = plt.subplots(nrows=1, ncols=3)
        # print(data_dict["voxels"].shape)
        # xyz = data_dict["voxels"].reshape(-1, 5)[:, 1:4]

        # ax[0].scatter(xyz[:, 0], xyz[:, 1], alpha=0.1, edgecolors='none')
        # ax[0].set_title("XY")
        # ax[0].grid(True)
        
        # ax[1].scatter(xyz[:, 0], xyz[:, 2], alpha=0.1, edgecolors='none')
        # ax[1].set_title("XZ")
        # ax[1].grid(True)

        # ax[2].scatter(xyz[:, 1], xyz[:, 2], alpha=0.1, edgecolors='none')
        # ax[2].set_title("YZ")
        # ax[2].grid(True)
        # plt.savefig(f"/workspace/level5_bags/voxelized_points_{suffix}.png", dpi=400, bbox_inches='tight')

        # # points
        # fig, ax = plt.subplots(nrows=1, ncols=3)
        # xyz = data_dict["points"][:, 1:4]
        # ax[0].scatter(xyz[:, 0], xyz[:, 1], alpha=0.1, edgecolors='none')
        # ax[0].set_title("XY")
        # ax[0].grid(True)
        
        # ax[1].scatter(xyz[:, 0], xyz[:, 2], alpha=0.1, edgecolors='none')
        # ax[1].set_title("XZ")
        # ax[1].grid(True)

        # ax[2].scatter(xyz[:, 1], xyz[:, 2], alpha=0.1, edgecolors='none')
        # ax[2].set_title("YZ")
        # ax[2].grid(True)
        # plt.savefig(f"/workspace/level5_bags/points_{suffix}.png", dpi=400, bbox_inches='tight')

        # # voxel_coords
        # fig, ax = plt.subplots(nrows=1, ncols=3)
        # xyz = data_dict["voxel_coords"][:, 1:4]
        # ax[0].scatter(xyz[:, 0], xyz[:, 1], alpha=0.1, edgecolors='none')
        # ax[0].set_title("voxel XY")
        # ax[0].grid(True)
        
        # ax[1].scatter(xyz[:, 0], xyz[:, 2], alpha=0.1, edgecolors='none')
        # ax[1].set_title("voxel XZ")
        # ax[1].grid(True)

        # ax[2].scatter(xyz[:, 1], xyz[:, 2], alpha=0.1, edgecolors='none')
        # ax[2].set_title("voxel YZ")
        # ax[2].grid(True)
        # plt.savefig(f"/workspace/level5_bags/voxel_coords_{suffix}.png", dpi=400, bbox_inches='tight')

        # # intensities
        # fig, ax = plt.subplots(nrows=1, ncols=2)
        # intensities = data_dict["points"][:, 4]
        # mask = np.logical_and(
        #     np.percentile(intensities, 1) < intensities,
        #     intensities < np.percentile(intensities, 99)
        # )
        # ax[0].hist(intensities[mask], bins=30)
        # ax[0].set_title("Intensity hist")
        # ax[0].grid(True)
        
        # voxels = data_dict["voxels"]
        # n_points_in_voxels = (voxels.sum(axis=-1) != 0).sum(axis=1)
        # mask = np.logical_and(
        #     np.percentile(n_points_in_voxels, 1) < n_points_in_voxels,
        #     n_points_in_voxels < np.percentile(n_points_in_voxels, 99)
        # )
        # ax[1].hist(n_points_in_voxels[mask], bins=10)
        # ax[1].set_title("Number points in voxels hist")
        # ax[1].grid(True)
        # plt.savefig(f"/workspace/level5_bags/intensity_points_in_voxels_{suffix}.png", dpi=400, bbox_inches='tight')
        # assert False

        load_data_to_gpu(data_dict)

        # TODO: add FP16 convertion
        with torch.no_grad():
            pred_dicts, _ = self.model(data_dict)
            print(pred_dicts[0]["pred_boxes"].shape)

        return pred_dicts
