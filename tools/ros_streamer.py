import argparse
import rospy
import ros_numpy
import message_filters

import numpy as np

from pcdet.config import cfg, cfg_from_yaml_file
from sensor_msgs.msg import PointCloud2

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from rviz_visualizer import RvizVisualizer
from streaming_detector import StreamingDetector

class Detection3DHandler:
    def __init__(self):
        rospy.init_node('pointcloud_detector', anonymous=True)

        self.args, self.cfg = self.parse_config()
        self.detector = StreamingDetector(self.cfg, self.args.ckpt)
        self.preds_publisher = None

        self.rviz_v = RvizVisualizer(header="base_link")
        # self.rviz_v = RvizVisualizer(header="lidar_top")
        self.rviz_publisher = rospy.Publisher('/detection_3d_markers', MarkerArray, queue_size=10)

        # top_lidar_sub = message_filters.Subscriber('/lidar_top', PointCloud2)
        # all_subs = [top_lidar_sub]

        top_lidar_sub = message_filters.Subscriber('/ouster_top_timestamped/points', PointCloud2)
        left_lidar_sub = message_filters.Subscriber('/ouster_left_timestamped/points', PointCloud2)
        right_lidar_sub = message_filters.Subscriber('/ouster_right_timestamped/points', PointCloud2)
        # rear_lidar_sub = message_filters.Subscriber('/lslidar_point_cloud', PointCloud2)
        all_subs = [top_lidar_sub, left_lidar_sub, right_lidar_sub]#, rear_lidar_sub]

        ts = message_filters.ApproximateTimeSynchronizer(
            all_subs,
            queue_size=10,
            slop=0.5 # почему так много?
        )
        ts.registerCallback(self.callback)

        try:
            rospy.spin()
        except KeyboardInterrupt:
            pass
    
    def parse_config(self):
        parser = argparse.ArgumentParser(description='arg parser')
        parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
        parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
        # TODO: add FP16 support 

        args = parser.parse_args()

        cfg_from_yaml_file(args.cfg_file, cfg)

        return args, cfg
            
    def transform_to_bin(self, *all_lidar_pc):
        numpified_pcs = []
        for idx in range(len(all_lidar_pc)):
            numpified_pcs.append(ros_numpy.numpify(all_lidar_pc[idx]))

        point_cloud = np.concatenate(numpified_pcs, axis=0)

        n_used_features = 5
        points = np.zeros((*point_cloud.shape, n_used_features))
    
        points[..., 0] = point_cloud['x']
        points[..., 1] = point_cloud['y']
        points[..., 2] = point_cloud['z']
        points[..., 3] = point_cloud['intensity']
        points[..., 4] = 0 # for timestamp

        points = np.array(points, dtype=np.float32).reshape(-1, n_used_features)
        print(points.shape)

        return points

    def rviz_update(self, pred_dict):
        self.rviz_publisher.publish(self.rviz_v.delete_all_markers())

        marker_array = MarkerArray()
        for idx in range(len(pred_dict["pred_boxes"])):
            bbox = pred_dict["pred_boxes"][idx]
            xyz_center = bbox[: 3]
            xyz_sizes = bbox[3: 6]
            angle = np.rad2deg(bbox[6].cpu())

            score = pred_dict["pred_scores"][idx]
            label = pred_dict["pred_labels"][idx]
            label_name = self.cfg.CLASS_NAMES[label - 1]

            cur_marker = self.rviz_v.get_marker(
                center=xyz_center,
                sizes=xyz_sizes,
                angles=(angle, 0, 0)
            )
            marker_array.markers.append(cur_marker)
            
            str_to_vis = f"{label_name} {score:.2f}"
            text_cur_marker = self.rviz_v.set_text_beside_marker(
                cur_marker, 
                text=str_to_vis
            )
            marker_array.markers.append(text_cur_marker)

        self.rviz_publisher.publish(marker_array)
    
    def send_predicts(self):
        pass

    def callback(self, *all_lidar_pc):
        points = self.transform_to_bin(*all_lidar_pc)
        
        pred_dicts = self.detector.predict_from_normalized_data(points)

        self.send_predicts()
        self.rviz_update(pred_dicts[0])

if __name__ == '__main__':
    Detection3DHandler()