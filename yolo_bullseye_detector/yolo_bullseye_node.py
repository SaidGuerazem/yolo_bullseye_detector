import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import os
from datetime import datetime
from pathlib import Path

class YOLODualDetector(Node):
    def __init__(self):
        super().__init__('yolo_dual_detector')
        model_path = Path(__file__).parent / "best_model.pt"
        # Load model on GPU
        self.model = YOLO(str(model_path)).to("cuda")
        self.get_logger().info(f"Model loaded on device: {next(self.model.model.parameters()).device}")

        self.bridge = CvBridge()
        self.output_dirs = {
            'FLIR': 'detections/FLIR',
            'CAM2': 'detections/CAM2'
        }

        for path in self.output_dirs.values():
            os.makedirs(path, exist_ok=True)

        self.create_subscription(Image, '/flir_camera/image_raw', self.image_callback_flir, 10)
        self.create_subscription(Image, '/cam_2/color/image_raw', self.image_callback_cam2, 10)

        self.pub_flir = self.create_publisher(String, '/flir_camera/centroids', 10)
        self.pub_cam2 = self.create_publisher(String, '/cam_2/centroids', 10)

    def image_callback_flir(self, msg):
        self.process_image(msg, self.pub_flir, 'FLIR')

    def image_callback_cam2(self, msg):
        self.process_image(msg, self.pub_cam2, 'CAM2')

    def process_image(self, msg, publisher, label):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            if msg.encoding.lower() == 'bayer_rggb8':
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BayerRG2BGR)
            elif msg.encoding.lower() == 'rgb8':
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            elif msg.encoding.lower() == 'mono8' or cv_image.ndim == 2:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)

            if cv_image is None or cv_image.size == 0:
                self.get_logger().warn(f"{label}: Empty image.")
                return

            resized_image = cv2.resize(cv_image, (640, 640))
            results = self.model(resized_image, verbose=False, conf=0.6)[0]
            obb = results.obb

            if obb is None or obb.xyxyxyxy is None or len(obb.xyxyxyxy) == 0:
                self.get_logger().warn(f"{label}: No OBB detections.")
                return

            polygons = obb.xyxyxyxy.cpu().numpy()
            confs = obb.conf.cpu().numpy() if obb.conf is not None else [0.0] * len(polygons)
            centroids = []

            for i, pts_flat in enumerate(polygons):
                pts = pts_flat.reshape((4, 2)).astype(np.int32)
                M = cv2.moments(pts)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Normalize centroid
                    norm_cx = cx / 640.0
                    norm_cy = cy / 640.0
                    centroids.append((norm_cx, norm_cy))

                    cv2.polylines(resized_image, [pts], True, (0, 255, 0), 2)
                    cv2.circle(resized_image, (cx, cy), 4, (0, 0, 255), -1)
                    label_text = f"bullseye {confs[i]:.2f}"
                    cv2.putText(resized_image, label_text, (pts[0][0], pts[0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    self.get_logger().warn(f"{label}: Degenerate polygon skipped.")

            # Save image with detections
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
            out_path = os.path.join(self.output_dirs[label], f"{label}_{timestamp}.jpg")
            cv2.imwrite(out_path, resized_image)

            # Publish centroids
            centroid_msg = f"{centroids}"
            publisher.publish(String(data=centroid_msg))
            self.get_logger().info(centroid_msg)

        except Exception as e:
            self.get_logger().error(f"Error in {label}: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = YOLODualDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

