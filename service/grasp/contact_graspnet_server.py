import argparse
import base64
import io
import os
import sys

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONTACT_GRASPNET_DIR = os.path.join(ROOT_DIR, "third_party", "contact_graspnet")
CONTACT_GRASPNET_PY_DIR = os.path.join(CONTACT_GRASPNET_DIR, "contact_graspnet")
DEFAULT_CHECKPOINT_DIR = os.path.join(
    CONTACT_GRASPNET_DIR,
    "checkpoints",
    "scene_test_2048_bs3_hor_sigma_001",
)

sys.path.insert(0, CONTACT_GRASPNET_PY_DIR)
sys.path.insert(0, CONTACT_GRASPNET_DIR)

import tensorflow.compat.v1 as tf

import config_utils
from contact_grasp_estimator import GraspEstimator


tf.disable_eager_execution()

for device in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)


app = Flask(__name__)
_MODEL = None


def decode_base64_rgb(base64_image: str) -> np.ndarray:
    image_data = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return np.array(image)


def decode_base64_depth(base64_image: str) -> np.ndarray:
    image_data = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image_data))
    depth_mm = np.array(image, dtype=np.uint32)
    return depth_mm.astype(np.float32) / 1000.0


def decode_base64_mask(base64_image: str) -> np.ndarray:
    image_data = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image_data))
    return np.array(image)


def to_serializable_array(value) -> list:
    if value is None:
        return []
    array = np.asarray(value)
    if array.size == 0:
        return []
    return array.tolist()


def select_prediction_group(pred_grasps_cam, scores, contact_pts, segmap_id: int):
    if segmap_id in pred_grasps_cam:
        return (
            pred_grasps_cam[segmap_id],
            scores.get(segmap_id, []),
            contact_pts.get(segmap_id, []),
        )

    if len(pred_grasps_cam) == 1:
        key = next(iter(pred_grasps_cam))
        return pred_grasps_cam[key], scores.get(key, []), contact_pts.get(key, [])

    grasp_batches = []
    score_batches = []
    contact_batches = []

    for key in sorted(pred_grasps_cam):
        grasp_array = np.asarray(pred_grasps_cam[key])
        score_array = np.asarray(scores.get(key, []))
        contact_array = np.asarray(contact_pts.get(key, []))

        if grasp_array.size == 0:
            continue

        grasp_batches.append(grasp_array)
        score_batches.append(score_array)
        contact_batches.append(contact_array)

    if not grasp_batches:
        return [], [], []

    return (
        np.concatenate(grasp_batches, axis=0),
        np.concatenate(score_batches, axis=0),
        np.concatenate(contact_batches, axis=0),
    )


def load_model():
    checkpoint_dir = os.environ.get("CONTACT_GRASPNET_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR)
    if not os.path.isdir(CONTACT_GRASPNET_DIR):
        raise FileNotFoundError(
            f"Contact-GraspNet checkout not found at {CONTACT_GRASPNET_DIR}. "
            "Run `pixi run clone-contact-graspnet` first."
        )
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(
            f"Checkpoint directory not found at {checkpoint_dir}. "
            "Download the pretrained Contact-GraspNet weights into third_party/contact_graspnet/checkpoints/."
        )

    forward_passes = int(os.environ.get("CONTACT_GRASPNET_FORWARD_PASSES", "1"))
    global_config = config_utils.load_config(checkpoint_dir, batch_size=forward_passes, arg_configs=[])

    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    saver = tf.train.Saver(save_relative_paths=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode="test")

    return {
        "checkpoint_dir": checkpoint_dir,
        "default_forward_passes": forward_passes,
        "grasp_estimator": grasp_estimator,
        "session": sess,
    }


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL


@app.route("/healthz", methods=["GET"])
def healthz():
    model = get_model()
    return jsonify(
        {
            "status": "ok",
            "checkpoint_dir": model["checkpoint_dir"],
        }
    )


@app.route("/sample_grasp", methods=["POST"])
def sample_grasp():
    data = request.get_json()
    rgb = decode_base64_rgb(data["image_rgb"])
    depth = decode_base64_depth(data["image_depth"])
    segmap = decode_base64_mask(data["segmap"])
    camera_matrix = np.array(data["K"], dtype=np.float32).reshape(3, 3)

    segmap_id = int(data.get("segmap_id", 0))
    z_range = data.get("z_range", [0.2, 1.8])
    local_regions = bool(data.get("local_regions", True))
    filter_grasps = bool(data.get("filter_grasps", True))
    skip_border_objects = bool(data.get("skip_border_objects", False))
    margin_px = int(data.get("margin_px", 5))

    model = get_model()
    forward_passes = int(data.get("forward_passes", model["default_forward_passes"]))

    pred_grasps_cam, scores, contact_pts, _ = model["grasp_estimator"].predict_scene_grasps_from_depth_K_and_2d_seg(
        model["session"],
        depth,
        segmap,
        camera_matrix,
        z_range=z_range,
        local_regions=local_regions,
        filter_grasps=filter_grasps,
        segmap_id=segmap_id,
        skip_border_objects=skip_border_objects,
        margin_px=margin_px,
        rgb=rgb,
        forward_passes=forward_passes,
    )

    grasp_values, score_values, contact_values = select_prediction_group(
        pred_grasps_cam,
        scores,
        contact_pts,
        segmap_id,
    )

    return jsonify(
        {
            "pred_grasps_cam": to_serializable_array(grasp_values),
            "scores": to_serializable_array(score_values),
            "contact_pts": to_serializable_array(contact_values),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contact-GraspNet server")
    parser.add_argument("--ip", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=4003, type=int)
    args = parser.parse_args()

    app.run(host=args.ip, port=args.port, debug=False, use_reloader=False)
