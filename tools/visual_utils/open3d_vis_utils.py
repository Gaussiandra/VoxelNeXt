"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import open3d.ml.torch as ml3d # мб добавить проверку?
import torch
import matplotlib
import numpy as np

box_colormap = [
    [0, 0, 1],
    [0, 1, 0],
    [1, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
    [0.5, 0.5, 0],
    [0.75, 0.75, 0],
    [0.4, 0.6, 0],
    [0, 0.2, 1],
    [0.5, 0.2, 1],
    [0.1, 0.3, 0.5]
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba



def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()


def get_data_for_ml3d(points, sample_index, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):  
    # дублирование
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()
    if isinstance(ref_labels, torch.Tensor):
        ref_labels = ref_labels.cpu().numpy()

    bboxes = []
    for idx in range(len(ref_boxes)):
        center = ref_boxes[idx, :3]
        size = ref_boxes[idx, 3:6]

        axis_angles = np.array([0, 0, ref_boxes[idx, 6] + 1e-10])
        rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)

        front = rot[:, 2]
        up = rot[:, 1]
        left = rot[:, 0]
        
        if ref_scores[idx] > 0.5:
            bbox = ml3d.vis.BoundingBox3D(
                center,
                front,
                up,
                left,
                size,
                label_class=ref_labels[idx],
                confidence=ref_scores[idx],
                arrow_length=0,
                show_class=True,
                show_confidence=True
            )  
            bboxes.append(bbox)
    
    data = {
        "name": f"points_{sample_index}",
        "points": points[:, :3],
        "intensity": np.clip(points[:, 3], 0, 40),
        "ring_index": points[:, 4],
        "bounding_boxes": bboxes,
        "labels": ref_labels # пофиксить
    }

    return data
    
def draw_scenes_ml3d(data_to_visualize):
    vis = ml3d.vis.Visualizer()

    # нужно связать с датасетом
    lut = ml3d.vis.LabelLUT()
    lut.add_label('1_str', 1)
    lut.add_label('2', 2)
    lut.add_label('3', 3)
    lut.add_label('4', 4)
    lut.add_label('5', 5)
    lut.add_label('6', 6)
    lut.add_label('7', 7)
    lut.add_label('8', 8)
    lut.add_label('9', 9)
    vis.set_lut("labels", lut)

    vis.visualize(data_to_visualize)
def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
 #           print(box_colormap)
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])
  #          print(ref_labels)
#            line_set.paint_uniform_color(color)

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis
