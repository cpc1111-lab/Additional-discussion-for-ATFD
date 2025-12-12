"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
Batch inference on a folder of images with parameters hardcoded.
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw

# ----------------------------
# �� 直接在代码中配置参数
# ----------------------------
CONFIG_PATH = "configs/dfine/custom/dfine_hgnetv2_l_custom.yml"
CHECKPOINT_PATH = "output1/dfine_hgnetv2_l_custom/best_stg1.pth"
INPUT_DIR = "/home/e222/cpc/D-FINE/dataset/test2017"
OUTPUT_DIR = "./results/det_vis1"
DEVICE = "cuda:0"
CONFIDENCE_THRESHOLD = 0.4
# ----------------------------

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig


def draw_and_save(image_pil, labels, boxes, scores, output_path, thrh=0.4):
    draw = ImageDraw.Draw(image_pil)

    scr = scores
    lab = labels[scr > thrh]
    box = boxes[scr > thrh]
    scrs = scr[scr > thrh]

    for j, b in enumerate(box):
        draw.rectangle(list(b), outline="red", width=2)
        draw.text(
            (b[0], b[1]),
            text=f"{lab[j].item()} {round(scrs[j].item(), 2)}",
            fill="blue",
        )

    image_pil.save(output_path)


def main():
    # Load config and model
    cfg = YAMLConfig(CONFIG_PATH, resume=CHECKPOINT_PATH)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = torch.device(DEVICE)
    model = Model().to(device).eval()

    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images. Starting inference...")

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    with torch.no_grad():
        for img_path in sorted(image_files):
            try:
                im_pil = Image.open(img_path).convert("RGB")
                w, h = im_pil.size
                orig_size = torch.tensor([[w, h]], device=device)

                im_data = transforms(im_pil).unsqueeze(0).to(device)
                labels, boxes, scores = model(im_data, orig_size)

                labels = labels[0]
                boxes = boxes[0]
                scores = scores[0]

                output_path = output_dir / f"{img_path.stem}_det{img_path.suffix}"
                draw_and_save(im_pil, labels, boxes, scores, output_path, thrh=CONFIDENCE_THRESHOLD)

                print(f"Processed: {img_path.name} → {output_path.name}")

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    print(f"\n✅ Inference complete. Results saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()