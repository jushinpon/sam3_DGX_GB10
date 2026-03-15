#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from pathlib import Path
from PIL import Image

#################################### For Image ####################################
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ============================================================
# 固定路徑設定
# ============================================================
SAM3_REPO = Path("/home/opt_arm/sam3")
BPE_PATH = SAM3_REPO / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
CHECKPOINT_DIR = SAM3_REPO / "checkpoints"

# 你的測試圖片
IMAGE_PATH = Path("./crowd.jpg")

# 輸出資料夾
OUTPUT_DIR = Path("./output")

# 自己定義 prompt
TEXT_PROMPT = "woman"

# 裝置設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_checkpoint(ckpt_dir: Path) -> Path:
    """
    自動尋找第一個 .pt checkpoint
    """
    candidates = sorted(ckpt_dir.rglob("*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint (.pt) found under: {ckpt_dir}\n"
            f"請先確認你已經把 SAM3 checkpoint 下載到這個路徑。"
        )
    return candidates[0]


def save_mask(mask_array: np.ndarray, out_path: Path) -> None:
    """
    將單一 mask 存成黑白 PNG
    """
    mask_img = Image.fromarray((mask_array.astype(np.uint8) * 255))
    mask_img.save(out_path)


def main():
    print("============================================================")
    print(" SAM3 image inference test")
    print("============================================================")
    print(f"Device         : {DEVICE}")
    print(f"SAM3 repo      : {SAM3_REPO}")
    print(f"BPE path       : {BPE_PATH}")
    print(f"Checkpoint dir : {CHECKPOINT_DIR}")
    print(f"Image path     : {IMAGE_PATH}")
    print(f"Text prompt    : {TEXT_PROMPT}")
    print("============================================================")

    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    if not BPE_PATH.is_file():
        raise FileNotFoundError(f"BPE file not found: {BPE_PATH}")

    checkpoint_path = find_checkpoint(CHECKPOINT_DIR)
    print(f"Checkpoint     : {checkpoint_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the model
    print(">>> Loading SAM3 model ...")
    model = build_sam3_image_model(
        bpe_path=str(BPE_PATH),
        device=DEVICE,
        checkpoint_path=str(checkpoint_path),
    )
    processor = Sam3Processor(model)

    # Load an image
    print(">>> Loading image ...")
    image = Image.open(IMAGE_PATH).convert("RGB")
    inference_state = processor.set_image(image)

    # Prompt the model with text
    print(">>> Running text prompt inference ...")
    output = processor.set_text_prompt(
        state=inference_state,
        prompt=TEXT_PROMPT
    )

    # Get the masks, bounding boxes, and scores
    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]

    # 轉成 numpy 方便後續處理
    if torch.is_tensor(masks):
        masks = masks.detach().cpu().numpy()
    else:
        masks = np.asarray(masks)

    if torch.is_tensor(boxes):
        boxes = boxes.detach().cpu().numpy()
    else:
        boxes = np.asarray(boxes)

    if torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()
    else:
        scores = np.asarray(scores)

    print("============================================================")
    print(" Inference Result")
    print("============================================================")
    print(f"Number of masks : {len(masks)}")
    print(f"Masks shape     : {masks.shape}")
    print(f"Boxes shape     : {boxes.shape}")
    print(f"Scores shape    : {scores.shape}")
    print("============================================================")

    for i in range(len(masks)):
        mask = masks[i]

        # 有些情況 mask shape 可能是 (1, H, W)
        if mask.ndim == 3:
            mask = mask[0]

        score = float(scores[i])
        box = boxes[i].tolist()

        print(f"[{i}] score = {score:.6f}, box = {box}")

        mask_path = OUTPUT_DIR / f"mask_{i:03d}.png"
        save_mask(mask > 0.5, mask_path)

    print("============================================================")
    print(f"All masks saved to: {OUTPUT_DIR}")
    print("Done.")
    print("============================================================")


if __name__ == "__main__":
    main()