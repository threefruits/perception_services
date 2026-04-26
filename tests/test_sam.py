#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apis.sam import SAM


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM smoke test")
    parser.add_argument("--server-url", default=os.environ.get("SAM_SERVER_URL", "http://127.0.0.1:4001"))
    parser.add_argument("--image", default=str(ROOT / "images" / "0.png"))
    args = parser.parse_args()

    segmentor = SAM(server_url=args.server_url)
    image = Image.open(args.image).convert("RGB")
    masks = segmentor.segment_auto_mask(image)

    print(f"SAM server: {args.server_url}")
    print(f"Image: {args.image}")
    print(f"Masks returned: {len(masks)}")
    if masks:
        first_mask = masks[0]["segmentation"]
        print(f"First mask shape: {first_mask.shape}")


if __name__ == "__main__":
    main()
