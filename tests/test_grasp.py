#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apis.grasp_estimator import GraspEstimator


def main() -> None:
    parser = argparse.ArgumentParser(description="Contact-GraspNet smoke test")
    parser.add_argument("--server-url", default=os.environ.get("GRASP_SERVER_URL", "http://127.0.0.1:4003"))
    parser.add_argument("--image", default=str(ROOT / "images" / "0.png"))
    args = parser.parse_args()

    rgb = np.array(Image.open(args.image).convert("RGB"))
    height, width = rgb.shape[:2]

    depth = np.full((height, width), 0.9, dtype=np.float32)
    segmap = np.ones((height, width), dtype=np.uint8)
    # Simple pinhole intrinsics guess for smoke testing.
    K = [550.0, 0.0, width / 2.0, 0.0, 550.0, height / 2.0, 0.0, 0.0, 1.0]

    estimator = GraspEstimator(server_url=args.server_url)
    pred_grasps_cam, scores, contact_pts = estimator.sample_grasp(
        image_rgb=rgb,
        image_depth=depth,
        segmap=segmap,
        K=K,
        segmap_id=1,
    )

    print(f"Grasp server: {args.server_url}")
    print(f"Image: {args.image}")
    print(f"Grasps: {len(pred_grasps_cam)}")
    print(f"Scores: {len(scores)}")
    print(f"Contact points: {len(contact_pts)}")


if __name__ == "__main__":
    main()
