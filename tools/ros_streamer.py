import time
import argparse
import rospy
import ros_numpy
import message_filters
import tf2_ros
import pickle 
import numpy as np
from collections import defaultdict
from scipy.spatial.transform import Rotation

from pcdet.config import cfg, cfg_from_yaml_file
from sensor_msgs.msg import PointCloud2

# from visualization_msgs.msg import Marker
# from visualization_msgs.msg import MarkerArray

# from rviz_visualizer import RvizVisualizer
from streaming_detector import StreamingDetector

from obstacle_detection_msgs.msg import Obstacle, ObstacleArray

class Detection3DHandler:
    label_to_type_map = {
        "person" : "Person",
        "pedestrian" : "Person",
        "car" : "Car",
        "bus" : "Car",
        "track" : "Car",
        "construction_vehicle" : "Car",
        "trailer" : "Car", 
        "motorcycle" : "Car",
        "bicycle" : "Car",
        "traffic_cone" : "traffic_cone",
        "barrier" : "barrier"
    }

    
    def __init__(self):
        rospy.init_node('voxelnet_detector', anonymous=True)

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.args, self.cfg = self.parse_config()
        self.detector = StreamingDetector(self.cfg, self.args.ckpt)
        self.preds_publisher = None

        self.oa_publisher = rospy.Publisher('/voxelnext/obstacles', ObstacleArray, queue_size=1) #10?
        self.pc_publisher = rospy.Publisher('/joined_point_cloud', PointCloud2, queue_size=1) #10?

        self.dump_points = False
        self.is_dumped = False
        self.dump_n_frames = 100
        self.points_to_dump = defaultdict(list)
        self.debug_on_nusc_bag = False
        self.lidar_names = ["top", "right", "left"]
        # self.lidar_names = ["top"]
        self.frame_id = "base_link"

        if self.debug_on_nusc_bag:
            top_lidar_sub = message_filters.Subscriber('/lidar_top', PointCloud2)
            all_subs = [top_lidar_sub]
            self.tRs_base_link_lidar = [(np.zeros(3), np.eye(3))]
        else:
            all_subs = []
            self.tRs_base_link_lidar = []
            for lidar_name in self.lidar_names:
                all_subs.append(message_filters.Subscriber(f'/ouster_{lidar_name}/points', PointCloud2))
                self.tRs_base_link_lidar.append(self.get_tvec_rot_mat_for_lidar(f"os_sensor_{lidar_name}"))
            
            all_subs.append(message_filters.Subscriber(f'/lslidar_point_cloud', PointCloud2))
            self.tRs_base_link_lidar.append(self.get_tvec_rot_mat_for_lidar(f"laser_link"))

        ts = message_filters.ApproximateTimeSynchronizer(
            all_subs,
            queue_size=1,
            slop=0.11
        )
        ts.registerCallback(self.callback)
        

        try:
            rospy.spin()
        except KeyboardInterrupt:
            pass
    
    def get_tvec_rot_mat_for_lidar(self, lidar_frame_id):
        tf_transform = self.tf_buffer.lookup_transform(
            self.frame_id, 
            lidar_frame_id, 
            rospy.Time(), 
            timeout=rospy.Duration(2)
        )
        translation = tf_transform.transform.translation
        rotation = tf_transform.transform.rotation

        t_btop = np.array([translation.x, translation.y, translation.z])
        R_btop = Rotation.from_quat([rotation.x,rotation.y, rotation.z, rotation.w])
        R_btop = R_btop.as_matrix().astype(float)

        return t_btop, R_btop
    
    def parse_config(self):
        parser = argparse.ArgumentParser(description='arg parser')
        parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
        parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
        # TODO: add FP16 support 

        args = parser.parse_args()

        cfg_from_yaml_file(args.cfg_file, cfg)

        return args, cfg
    
    def translate_and_rotate(self, points, t, R):
        if self.debug_on_nusc_bag:
            return points

        point_base_link = points @ R.T + t

        return  point_base_link

    def transform_to_bin(self, *all_lidar_pc):
        points_list = []
        
        for idx in range(len(all_lidar_pc)):
            point_cloud = ros_numpy.numpify(all_lidar_pc[idx])

            n_used_features = 3
            points = np.zeros((*point_cloud.shape, n_used_features))

            points[..., 0] = point_cloud['x']
            points[..., 1] = point_cloud['y']
            points[..., 2] = point_cloud['z']
            # points[..., 3] = point_cloud['intensity']
            # points[..., 4] = 0 # for timestamp

            points = np.array(points, dtype=np.float32).reshape(-1, n_used_features)
            Rs = np.linalg.norm(points[:, :3], axis=1)
            # points = points[(Rs < 55) & (Rs > 3) & (points[:, 2] < 5)]
            # points = points[(points[:, 0] > 0.001)&((points[:, 0] < -0.001))]
            
            points[:, :3] = self.translate_and_rotate(
                points[:, :3], 
                *self.tRs_base_link_lidar[idx]
            )
            points = points[(points[:, 2] < 3.2)&(Rs < 50) & (Rs > 2)]
            # points[:, 3] *= 0.1464823
            points_list.append(points)

            if self.dump_points and not self.is_dumped:
                lidar_name = self.lidar_names[idx]

                if len(self.points_to_dump[lidar_name]) < self.dump_n_frames:
                    self.points_to_dump[lidar_name].append(points)
                print(lidar_name, len(self.points_to_dump[lidar_name]))

        if self.dump_points and not self.is_dumped:
            is_ready = [len(v) == self.dump_n_frames for v in self.points_to_dump.values()]
            if all(is_ready):
                self.is_dumped = True
                pickle.dump(self.points_to_dump, open("np_points.pkl", "wb"))
                print("points dumped")

        points = np.concatenate(points_list)

        return points
    
    
    
    def map_label_to_type(self, label):
        return self.label_to_type_map.get(label, "")

    def send_predicts(self, pred_dict, stamp):

        obstacles = ObstacleArray()
        obstacles.header.frame_id = self.frame_id
        obstacles.header.stamp = stamp

        for idx in range(len(pred_dict["pred_boxes"])):
            bbox = pred_dict["pred_boxes"][idx]
            xyz_center = bbox[: 3]
            xyz_sizes = bbox[3: 6]
            angle = bbox[6].cpu()

            score = pred_dict["pred_scores"][idx]
            label = pred_dict["pred_labels"][idx]
            label_name = self.cfg.CLASS_NAMES[label - 1]
            obs = Obstacle()
            ty = self.map_label_to_type(label_name)
            if ty != "Car" and ty != "Person":
                continue
            obs.type = ty
            obs.confidence = score
            obs.scale.x = xyz_sizes[0]
            obs.scale.y = xyz_sizes[1]
            obs.scale.z = xyz_sizes[2]
            orientation = Rotation.from_euler("z", angle, degrees=False).as_quat()
            obs.pose.orientation.x = orientation[0]
            obs.pose.orientation.y = orientation[1]
            obs.pose.orientation.z = orientation[2]
            obs.pose.orientation.w = orientation[3]
            obs.pose.position.x = xyz_center[0]
            obs.pose.position.y = xyz_center[1]
            obs.pose.position.z = xyz_center[2]
            
            obstacles.obstacles.append(obs)
            

        self.oa_publisher.publish(obstacles)
    

    def publish_joined_point_cloud(self, points, stamp):
        data = np.zeros(points.shape[0], dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            # ('intensity', np.float32)
        ])
        data['x'] = points[:, 0]
        data['y'] = points[:, 1]
        data['z'] = points[:, 2]
        # data['intensity'] = points[:, 3]

        msg = ros_numpy.msgify(PointCloud2, data)
        msg.header.frame_id="base_link"
        msg.header.stamp = stamp # ????
        self.pc_publisher.publish(msg)

    def callback(self, *all_lidar_pc):
        t = time.time()
        stamp = rospy.Time.now()
        if (stamp - all_lidar_pc[0].header.stamp) > rospy.Duration(0.2):
            print("PAST", stamp,  all_lidar_pc[0].header.stamp)
            return
        print("all_lidar_pc: ", *[m.header for m in all_lidar_pc])
        print("stamp", stamp)
        t1 = time.time()
        points = self.transform_to_bin(*all_lidar_pc)
        print("transform_to_bin time", time.time() - t1)
        t1 = time.time()
        pred_dicts = self.detector.predict_from_normalized_data(points)
        print("predict from nomolized time", time.time() - t1)
        self.publish_joined_point_cloud(points, stamp)

        self.send_predicts(pred_dicts[0], stamp)
        # time.sleep(0.2)
        print("all", time.time() - t)

if __name__ == '__main__':
    Detection3DHandler()
