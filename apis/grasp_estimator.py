import requests
import pickle
import base64
import os
import numpy as np
from PIL import Image

import io
import random
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import base64
from io import BytesIO
# from visualization_utils import visualize_grasps, show_image

def convert_pil_image_to_base64(image: Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


class GraspEstimator():
    def __init__(self, server_url=None):
        self.server_url = server_url or os.environ.get("GRASP_SERVER_URL", "http://127.0.0.1:4003")

    def sample_grasp(self, image_rgb: np.ndarray, image_depth: np.ndarray, segmap: np.ndarray, K: list, segmap_id: int):
        """
        Send a request to the server with the specified image and additional data.
        
        :param endpoint: The endpoint for the specific segmentation method.
        :param image: The image to be segmented.
        :param additional_data: Additional data required by the specific method.
        :return: pred_grasps_cam, scores, contact_pts
        """
        image_depth = np.nan_to_num(image_depth)
        image_depth = image_depth*1000
        image_depth = image_depth.astype(np.uint32)

        image_rgb = Image.fromarray(image_rgb)
        image_depth = Image.fromarray(image_depth)
        

        segmap = Image.fromarray(segmap)
        image_rgb_base64 = convert_pil_image_to_base64(image_rgb)
        image_depth_base64 = convert_pil_image_to_base64(image_depth)
        segmap_base64 = convert_pil_image_to_base64(segmap)
        payload = {"image_rgb": image_rgb_base64, "image_depth": image_depth_base64, "segmap": segmap_base64, "K": K, "segmap_id": segmap_id}

        # Convert numpy arrays to lists
        for key, value in payload.items():
            if isinstance(value, np.ndarray):
                payload[key] = value.tolist()

        response = requests.post(f"{self.server_url}/sample_grasp", json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors

        return response.json()['pred_grasps_cam'], response.json()['scores'], response.json()['contact_pts']
    
    def visualize_grasp(self, pred_grasps_cam, depth, K, top_grasp_idx=None, rgb=None, z_range=[0.2,1.8], skip_border_objects=False, view_params=None):
        from .utils import extract_point_clouds
        import open3d as o3d

        cam_K = np.array(K).reshape(3,3)
        pc_full, pc_segments, pc_colors = extract_point_clouds(depth, cam_K, rgb=rgb, skip_border_objects=skip_border_objects, z_range=z_range)
        
        # Create point cloud visualization
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_full)
        if rgb is not None:
            pc_colors = pc_colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(pc_colors)

        # Create gripper visualization for each grasp
        gripper_geometries = []
        # only visualize the top 10 grasps
        # rank grasps by score
        # sorted_grasps = sorted(zip(pred_grasps_cam, scores), key=lambda x: x[1], reverse=True)

        for i, grasp in enumerate(pred_grasps_cam):
            grasp_mat = np.array(grasp).reshape(4, 4)
            
            # Define gripper dimensions (in meters)
            width = 0.12  # Gripper width
            depth = 0.1  # Gripper depth
            line_width = 0.02

            # Define 6 control points for gripper in local coordinates
            # Two points for base, four points for fingers
            points = [
                [0, 0, 0],  # center of the gripper
                [width/2, 0, 0],  # right base
                [-width/2, 0, 0],   # left base
                [0, 0, -0.06],  # handle
                [width/2, 0, depth],  # Right finger front
                [-width/2, 0, depth],  # Left finger front
            ]
            
            # Convert points to homogeneous coordinates and transform
            points = np.array(points)
            points_homog = np.concatenate([points, np.ones((6, 1))], axis=1)
            transformed_points = (grasp_mat @ points_homog.T).T[:, :3]
            
            # Create lines connecting the points
            lines = [
                [0, 1],  # Right base
                [0, 2],  # Left base
                [0, 3],  # handle
                [1, 4],  # Right finger
                [2, 5],  # Left finger
            ]
            # change line width
            # Create LineSet for the gripper
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(transformed_points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            
            # Color the lines based on whether it's the top grasp
            colors = [[1, 0, 0] for _ in range(len(lines))]  # Red color
            if top_grasp_idx is not None:
                if i == top_grasp_idx:  # Top scoring grasp
                    colors = [[0, 0, 1] for _ in range(len(lines))]  # Blue color
                else:
                    colors = [[1, 0, 0] for _ in range(len(lines))]  # Red color
            line_set.colors = o3d.utility.Vector3dVector(colors)
            
            gripper_geometries.append(line_set)

        # Create visualization window with custom view
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # Add geometries
        vis.add_geometry(pcd)
        for gripper in gripper_geometries:
            vis.add_geometry(gripper)
        # Set default view parameters if none provided
        if view_params is None:
            view_params = {
                'front': [0, 0, -1],  # Looking towards negative z
                'lookat': [0, 0, 0],  # Looking at origin
                'up': [0, -1, 0],     # Y-axis points up
                'zoom': 0.7
            }
        
        # Get and setup the view control
        ctr = vis.get_view_control()
        ctr.set_front(view_params['front'])
        ctr.set_lookat(view_params['lookat'])
        ctr.set_up(view_params['up'])
        ctr.set_zoom(view_params['zoom'])
        
        # Run visualization
        vis.run()
        vis.destroy_window()



if __name__ == "__main__":
    grasp_estimator = GraspEstimator()
    # generate fake data

    image_rgb = Image.new("RGB", (100, 100), "white")
    image_depth = Image.new("L", (100, 100), 0)
    segmap = Image.new("L", (100, 100), 0)
    K = [570.3, 0, 320, 0, 570.3, 240, 0, 0, 1]
    segmap_id = 1

    res = grasp_estimator.sample_grasp(image_rgb, image_depth, segmap, K, segmap_id)
    # print(res)
    
