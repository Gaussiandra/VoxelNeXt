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
        # print("voxels[0]", data_dict["voxels"][0])
        # print("voxel_coords", data_dict["voxel_coords"].shape)
        # print("voxel_num_points", data_dict["voxel_num_points"].shape)
        # print("batch_size", data_dict["batch_size"])

        load_data_to_gpu(data_dict)

        # TODO: add FP16 convertion
        with torch.no_grad():
            pred_dicts, _ = self.model(data_dict)
            print(pred_dicts[0]["pred_boxes"].shape)

        return pred_dicts
