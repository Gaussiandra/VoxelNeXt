import torch
import numpy as np
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import matplotlib.pyplot as plt
import time
try:
    import kornia
except:
    pass 

HALF = True

def load_data_to_gpu_half(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous().half()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda().half()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda().half()

class StreamingDetector:
    def __init__(self, cfg, ckpt_path):
        self.logger = common_utils.create_logger(log_file="./temp_log.log")
        
        cfg.DATA_CONFIG.POINT_FEATURE_ENCODING = {
            "encoding_type": "absolute_coordinates_encoding",
            "used_feature_list": ['x', 'y', 'z'],
            "src_feature_list": ['x', 'y', 'z'],
        }

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
        if HALF:
            self.model.half()

    def predict_from_normalized_data(self, points):
        data_dict = {
            'points': points,
            'batch_size': 1
        }

        # print("points before prepare_data", data_dict["points"])
        t = time.time()
        data_dict = self.dataset.prepare_data(data_dict=data_dict)
        data_dict = self.dataset.collate_batch([data_dict])
        print(np.mean(points, axis=0), np.mean(data_dict["points"], axis=0))
        print("prep + collate", time.time() - t)
        t = time.time()
        
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
        if HALF:
            load_data_to_gpu_half(data_dict)
        else:
            load_data_to_gpu(data_dict)
        print("load", time.time() - t)
        t = time.time()
        # TODO: add FP16 convertion

        with torch.no_grad():
            pred_dicts, _ = self.model(data_dict)
            print(pred_dicts[0]["pred_boxes"].shape)
        print("model", time.time() - t)
        # t = time.time()
        return pred_dicts
