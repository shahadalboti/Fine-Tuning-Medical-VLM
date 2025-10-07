import os
import re
import cv2
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
from transformers import AutoProcessor, AutoModelForImageTextToText

# ---------- Image preprocessing (CLAHE + optional resize) ----------
class ApplyCLAHE:
    def __init__(self, cliplimit=1.0, tilesize=8):
        self.cliplimit = cliplimit
        self.tilesize = tilesize
    def __call__(self, image):
        image = np.array(image)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.cliplimit, tileGridSize=(self.tilesize, self.tilesize))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        image_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(image_clahe)

def load_rgb(image_path):
    img = Image.open(image_path).convert("RGB")
    return img

# ---------- Generation helpers ----------
def generate_response(model, processor, messages, max_new_tokens=100, temperature=0.3, max_retries=2):
    for attempt in range(max_retries):
        try:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True
                )
            generated_text = generated_ids[0][input_len:]
            return processor.decode(generated_text, skip_special_tokens=True).strip()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    return "Error: Unable to generate response"

def parse_response_advanced(response):
    response_lower = response.lower()
    is_fundus = "Unknown"
    has_dr = "Unknown"

    if any(p in response_lower for p in ["fundus': 'yes", "fundus: yes", "is a fundus", "valid fundus", "retinal fundus", "fundus yes", "fundus:yes"]):
        is_fundus = "Fundus"
    elif any(p in response_lower for p in ["fundus': 'no", "fundus: no", "not a fundus", "not fundus", "non-fundus", "fundus no", "fundus:no"]):
        is_fundus = "Not Fundus"

    if is_fundus == "Fundus":
        if any(p in response_lower for p in ["diagnosis': 'dr", "diagnosis: dr", "diabetic retinopathy", "signs of dr", "has dr", "dr_positive", "dr positive"]):
            has_dr = "DR"
        elif any(p in response_lower for p in ["diagnosis': 'no dr", "diagnosis: no dr", "no diabetic retinopathy", "no signs", "no dr", "dr_negative", "dr negative"]):
            has_dr = "No DR"

    if is_fundus == "Unknown":
        if "fundus" in response_lower and "not" not in response_lower.split("fundus")[0][-20:]:
            is_fundus = "Fundus"
        elif "fundus" in response_lower:
            is_fundus = "Not Fundus"

    if has_dr == "Unknown" and is_fundus == "Fundus":
        if "dr" in response_lower and "no" not in response_lower.split("dr")[0][-10:]:
            has_dr = "DR"
        elif "dr" in response_lower or "diabetic" in response_lower:
            has_dr = "No DR"

    return is_fundus, has_dr

def build_fundus_prompt(image):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are an expert ophthalmologist. Your only task is to determine if the given image is a valid retinal fundus photograph.\n"
                        "A fundus image should show:\n"
                        "- The retina with visible blood vessels\n"
                        "- Optic disc (bright circular area)\n"
                        "- Macula (darker central area)\n"
                        "- Circular field of view typical of fundus photography\n"
                        'Respond ONLY with: "FUNDUS" or "NOT_FUNDUS"\nDo not provide any additional explanation.'
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Is this image a valid retinal fundus photograph? Answer with FUNDUS or NOT_FUNDUS only."},
                {"type": "image", "image": image},
            ],
        },
    ]

def build_dr_prompt(image):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are an expert retinal specialist. This image has been confirmed as a retinal fundus photograph.\n"
                        "Your task is to detect signs of diabetic retinopathy (DR) including:\n"
                        "- Microaneurysms, Hemorrhages, Hard/Soft exudates, Neovascularization, Venous beading, IRMA\n"
                        'Respond ONLY with: "DR_POSITIVE" if you see any signs, or "DR_NEGATIVE" if normal.\nDo not provide any additional explanation.'
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Does this fundus image show any signs of diabetic retinopathy? Answer with DR_POSITIVE or DR_NEGATIVE only."},
                {"type": "image", "image": image},
            ],
        },
    ]

def build_fallback_prompt(image):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are an expert ophthalmologist. Analyze this medical image carefully.\n"
                        "First: Is this a retinal fundus photograph?\n"
                        "Second: If yes, does it show diabetic retinopathy?\n"
                        "Respond in this exact format:\nFUNDUS: [YES/NO]\nDIAGNOSIS: [DR/NO_DR/UNKNOWN]"
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image for fundus classification and diabetic retinopathy detection."},
                {"type": "image", "image": image},
            ],
        },
    ]

def main():
    parser = argparse.ArgumentParser(description="Med-GEMMA inference with double prompt for Fundus + DR detection")
    parser.add_argument("--model_id", type=str, default="google/medgemma-27b-it", help="Base model id or path")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory (HF cache)")
    parser.add_argument("--local_files_only", action="store_true", help="Use only local files (no internet)")
    parser.add_argument("--image_root", type=str, required=True, help="Root folder of images")
    parser.add_argument("--ground_truth_csv", type=str, required=True, help="CSV with 'Image' and 'label' columns")
    parser.add_argument("--output_csv", type=str, required=True, help="Where to save merged results")
    parser.add_argument("--apply_clahe", action="store_true", help="Apply CLAHE")
    parser.add_argument("--clahe_cliplimit", type=float, default=1.0, help="CLAHE clip limit")
    parser.add_argument("--clahe_tilesize", type=int, default=8, help="CLAHE tile size")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max new tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    args = parser.parse_args()

    # Load model & processor
    print("🧠 Loading model and processor...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
        device_map="auto",
        local_files_only=args.local_files_only,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )

    # Data
    gt_df = pd.read_csv(args.ground_truth_csv, encoding="utf-8")
    image_list = gt_df["Image"].dropna().astype(str).tolist()

    # Preproc
    clahe = ApplyCLAHE(cliplimit=args.clahe_cliplimit, tilesize=args.clahe_tilesize) if args.apply_clahe else None #Turn Off CLAHE for faster processing
    label_map = {"Fundus": 1, "Not Fundus": 0, "DR": 1, "No DR": 0, "Unknown": -1}

    results = []
    for filename in tqdm(image_list, desc="🔍 Evaluating Med-GEMMA with Double Prompt"):
        if not isinstance(filename, str) or filename.lower() in ["nan", "none", "", "null"]:
            print(f"Skipping invalid filename: {filename}")
            continue
        image_path = os.path.join(args.image_root, filename)
        if not os.path.exists(image_path):
            print(f"File not found: {filename}")
            continue

        try:
            image = load_rgb(image_path)
            if clahe: image = clahe(image) #Turn Off CLAHE for faster processing
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        # 1) Fundus prompt
        fundus_messages = build_fundus_prompt(image)
        fundus_response = generate_response(
            model, processor, fundus_messages,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature
        )

        is_fundus = "Not Fundus"
        fr = fundus_response.lower()
        if "fundus" in fr and "not" not in fr:
            is_fundus = "Fundus"

        has_dr = "Unknown"
        dr_response = "N/A - Not a fundus image"

        # 2) DR prompt (if fundus)
        if is_fundus == "Fundus":
            dr_messages = build_dr_prompt(image)
            dr_response = generate_response(
                model, processor, dr_messages,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature
            )
            drl = dr_response.lower()
            if "positive" in drl:
                has_dr = "DR"
            elif "negative" in drl:
                has_dr = "No DR"
            else:
                if "dr" in drl and "no" not in drl:
                    has_dr = "DR"
                else:
                    has_dr = "No DR"

        # 3) Fallback combined
        if is_fundus == "Unknown" or (is_fundus == "Fundus" and has_dr == "Unknown"):
            fallback_messages = build_fallback_prompt(image)
            fallback_response = generate_response(
                model, processor, fallback_messages,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature
            )
            fb_fundus, fb_dr = parse_response_advanced(fallback_response)
            if is_fundus == "Unknown":
                is_fundus = fb_fundus
            if has_dr == "Unknown":
                has_dr = fb_dr

        results.append({
            "Image": filename,
            "fundus_response": fundus_response,
            "dr_response": dr_response,
            "is_fundus": is_fundus,
            "has_dr": has_dr,
            "is_fundus_int": label_map.get(is_fundus, -1),
            "has_dr_int": label_map.get(has_dr, -1),
        })

    df = pd.DataFrame(results)
    print("\n📊 Classification Statistics:")
    print(f"Fundus classifications: {df['is_fundus'].value_counts().to_dict()}")
    print(f"DR classifications: {df['has_dr'].value_counts().to_dict()}")
    print(f"Unknown predictions: {(df['has_dr'] == 'Unknown').sum()}")

    # Merge and metrics
    df = df.merge(gt_df, on="Image", how="left")
    gt_labels = df["label"].fillna(-1).astype(int)
    pred_labels = df["has_dr_int"]
    valid_mask = (pred_labels != -1) & (gt_labels != -1)
    y_true = gt_labels[valid_mask]
    y_pred = pred_labels[valid_mask]

    if len(y_true) > 0:
        try:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                uniq = np.unique(np.concatenate([y_true, y_pred]))
                if len(uniq) == 1:
                    if uniq[0] == 0:
                        tn, fp, fn, tp = len(y_true), 0, 0, 0
                    else:
                        tn, fp, fn, tp = 0, 0, 0, len(y_true)
                else:
                    tn, fp, fn, tp = (0, 0, 0, 0)
        except Exception as e:
            print(f"Error computing confusion matrix: {e}")
            tn, fp, fn, tp = 0, 0, 0, 0

        acc = accuracy_score(y_true, y_pred)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print("\nEvaluation Metrics:")
        print(f" Accuracy: {acc:.4f}")
        print(f" Sensitivity (DR): {sensitivity:.4f}")
        print(f" Specificity (No DR): {specificity:.4f}")
        print(f" Confusion Matrix - TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        print(f" Total valid predictions: {len(y_true)}")
        print(f" Unknown predictions excluded: {(pred_labels == -1).sum()}")
    else:
        print(" No valid predictions matched with ground truth.")

    # Save
    try:
        df.to_csv(args.output_csv, index=False)
        print(f"\n Final results saved to: {args.output_csv}")
        print(f" Total processed images: {len(df)}")
    except Exception as e:
        print(f" Error saving results: {e}")

    print("\n Double prompt processing completed!")

if __name__ == "__main__":
    main()
