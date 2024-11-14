from apis.grasp_estimator import GraspEstimator
from PIL import Image
import rospy
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
import numpy as np
import time
import glob



class GraspEstimatorNode:
    def __init__(self):
        rospy.init_node('grasp_estimator_node')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Initialize grasp estimator
        self.grasp_estimator = GraspEstimator()
        
        # Camera intrinsics (adjust these values for your camera)
        self.K = [554.254691191187, 0.0, 320.5, 0.0, 554.254691191187, 240.5, 0.0, 0.0, 1.0]
        
        # Subscribe to topics
        self.rgb_sub = rospy.Subscriber('/head_camera/rgb/image_raw', RosImage, self.rgb_callback)
        self.depth_sub = rospy.Subscriber('/head_camera/depth_registered/image_raw', RosImage, self.depth_callback)
        
        # Store latest messages
        self.latest_rgb = None
        self.latest_depth = None


    def rgb_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            self.latest_rgb = cv_image
            
        except Exception as e:
            rospy.logerr(f"Error processing RGB image: {e}")

    def depth_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg)
            self.latest_depth = cv_image
        except Exception as e:
            rospy.logerr(f"Error processing depth image: {e}")

    def estimate_grasp(self):
        if self.latest_rgb is not None and self.latest_depth is not None:
            try:
                # Create segmap of ones with same size as RGB image
                segmap = np.ones((self.latest_rgb.shape[0], self.latest_rgb.shape[1]), dtype=np.uint8)
                segmap_id = 0  # Adjust this based on your segmentation map
                pred_grasps_cam, scores, contact_pts = self.grasp_estimator.sample_grasp(
                    self.latest_rgb,
                    self.latest_depth,
                    segmap,
                    self.K,
                    segmap_id
                )
                self.grasp_estimator.visualize_grasp(pred_grasps_cam, self.latest_depth, self.K, rgb=self.latest_rgb, top_grasp_idx=0)

                # rospy.loginfo(f"Grasp estimation result: {result}")
            except Exception as e:
                rospy.logerr(f"Error estimating grasp: {e}")

if __name__ == '__main__':

    node = GraspEstimatorNode()
    # segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(p, K=K)


    time.sleep(1)
    start_time = time.time()
    node.estimate_grasp()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

