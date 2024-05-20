from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from scipy.spatial.transform import Rotation

class RvizVisualizer:
    def __init__(self, header):
        self.header = header

        self.cur_marker_max_idx = -1

        self.__delete_all_arr = MarkerArray()
        m = Marker()
        m.action = Marker.DELETEALL
        # заменить на latency
        self.__delete_all_arr.markers.append(m)
    
    def delete_all_markers(self):
        self.cur_marker_max_idx = -1
        return self.__delete_all_arr

    def get_marker(self, center, sizes, angles, class_idx=None, proba=None):
        marker = Marker()
        marker.header.frame_id = self.header
        marker.type = marker.CUBE
        marker.action = marker.ADD

        marker.pose.position.x = center[0]
        marker.pose.position.y = center[1]
        marker.pose.position.z = center[2]

        marker.scale.x = sizes[0]
        marker.scale.y = sizes[1]
        marker.scale.z = sizes[2]

        # TODO: добавить выбор цвета в зависимоти от класса
        marker.color.a = 0.5
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        
        orientation = Rotation.from_euler("z", angles[0], degrees=True).as_quat()
        marker.pose.orientation.x = orientation[0]
        marker.pose.orientation.y = orientation[1]
        marker.pose.orientation.z = orientation[2]
        marker.pose.orientation.w = orientation[3]

        self.cur_marker_max_idx += 1
        marker.id = self.cur_marker_max_idx

        return marker
    
    def set_text_beside_marker(self, key_marker, text, text_size=1):
        marker = Marker()

        marker.type = marker.TEXT_VIEW_FACING
        marker.text = text

        marker.header.frame_id = key_marker.header.frame_id
        marker.action = marker.ADD
        marker.pose = key_marker.pose
        marker.scale.z = text_size

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0

        self.cur_marker_max_idx += 1
        marker.id = self.cur_marker_max_idx

        return marker