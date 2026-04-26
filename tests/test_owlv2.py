#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apis.owlv2 import OWLViT


def main() -> None:
    parser = argparse.ArgumentParser(description="OWLv2 smoke test")
    parser.add_argument("--server-url", default=os.environ.get("OWLV2_SERVER_URL", "http://127.0.0.1:4000"))
    parser.add_argument("--image", default=str(ROOT / "images" / "0.png"))
    parser.add_argument("--query", nargs="+", default=["cup", "bottle"])
    parser.add_argument("--threshold", type=float, default=0.12)
    args = parser.parse_args()

    detector = OWLViT(server_url=args.server_url)
    image = Image.open(args.image).convert("RGB")

    result = detector.detect_objects(
        image=image,
        text_queries=args.query,
        bbox_score_top_k=20,
        bbox_conf_threshold=args.threshold,
    )

    print(f"OWLv2 server: {args.server_url}")
    print(f"Image: {args.image}")
    print(f"Queries: {args.query}")
    print(f"Detections: {len(result)}")
    if result:
        print(f"Top result: {result[0]}")


if __name__ == "__main__":
    main()
