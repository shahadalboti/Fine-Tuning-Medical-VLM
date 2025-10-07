import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

def apply_clahe_rgb(img: Image.Image, cliplimit: float = 1.0, tilesize: int = 8) -> Image.Image:
    import cv2
    arr = np.array(img)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(tilesize, tilesize))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(out)

def build_msgs_fundus(image):
    return [
        {"role": "system","content":[{"type":"text","text": "You are an expert ophthalmologist. Respond ONLY with FUNDUS or NOT_FUNDUS."}]},
        {"role":"user","content":[{"type":"text","text":"Is this image a valid retinal fundus photograph? FUNDUS or NOT_FUNDUS only."},{"type":"image","image":image}]}
    ]

def build_msgs_dr(image):
    return [
        {"role":"system","content":[{"type":"text","text":"You are an expert retinal specialist. Respond ONLY with DR_POSITIVE or DR_NEGATIVE."}]},
        {"role":"user","content":[{"type":"text","text":"Does this fundus image show diabetic retinopathy? DR_POSITIVE or DR_NEGATIVE only."},{"type":"image","image":image}]}
    ]

def build_msgs_fallback(image):
    return [
        {"role":"system","content":[{"type":"text","text":"FUNDUS: [YES/NO]\nDIAGNOSIS: [DR/NO_DR/UNKNOWN]"}]},
        {"role":"user","content":[{"type":"text","text":"Analyze for fundus and DR."},{"type":"image","image":image}]}
    ]

def generate(model, processor, msgs, max_new_tokens=64, temperature=0.0):
    text = processor.apply_chat_template(msgs, tokenize=False)
    vis = process_vision_info(msgs)[0]
    inputs = processor(text=[text], images=[vis], return_tensors="pt")
    inputs = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k,v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=(temperature>0))
    return processor.decode(out[0], skip_special_tokens=True)

def parse_fallback(resp: str):
    low = resp.lower()
    fundus = "Fundus" if "fundus: yes" in low else ("Not Fundus" if "fundus: no" in low else "Unknown")
    if "diagnosis: dr" in low and "no_dr" not in low and "no dr" not in low:
        dr = "DR"
    elif "diagnosis: no_dr" in low or "diagnosis: no dr" in low:
        dr = "No DR"
    else:
        dr = "Unknown"
    return fundus, dr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="lingshu-medical-mllm/Lingshu-32B")
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--ground_truth_csv", type=str, required=True)
    ap.add_argument("--output_csv", type=str, required=True)
    ap.add_argument("--apply_clahe", action="store_true")
    ap.add_argument("--clahe_cliplimit", type=float, default=1.0)
    ap.add_argument("--clahe_tilesize", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto",
        cache_dir=args.cache_dir, local_files_only=args.local_files_only,
    )
    processor = Qwen2VLProcessor.from_pretrained(
        args.model_id, cache_dir=args.cache_dir, use_fast=True,
        min_pixels=256*28*28, max_pixels=1280*28*28
    )

    gt = pd.read_csv(args.ground_truth_csv)
    image_list = gt["Image"].dropna().astype(str).tolist()

    results = []
    for fname in tqdm(image_list, desc="🔍 Lingshu Double Prompt"):
        p = os.path.join(args.image_root, fname)
        if not os.path.exists(p):
            print(f"Missing: {fname}")
            continue
        img = Image.open(p).convert("RGB")
        if args.apply_clahe:
            img = apply_clahe_rgb(img, args.clahe_cliplimit, args.clahe_tilesize)

        fr = generate(model, processor, build_msgs_fundus(img), args.max_new_tokens, args.temperature)
        is_fundus = "Fundus" if ("fundus" in fr.lower() and "not" not in fr.lower()) else "Not Fundus"

        has_dr = "Unknown"; dr_text = "N/A"
        if is_fundus == "Fundus":
            dr_text = generate(model, processor, build_msgs_dr(img), args.max_new_tokens, args.temperature)
            low = dr_text.lower()
            if "dr_positive" in low or ("dr" in low and "no" not in low):
                has_dr = "DR"
            elif "dr_negative" in low or "no dr" in low or "no_dr" in low:
                has_dr = "No DR"

        if is_fundus == "Unknown" or (is_fundus == "Fundus" and has_dr == "Unknown"):
            fb = generate(model, processor, build_msgs_fallback(img), args.max_new_tokens, args.temperature)
            fb_fundus, fb_dr = parse_fallback(fb)
            if is_fundus == "Unknown":
                is_fundus = fb_fundus
            if has_dr == "Unknown":
                has_dr = fb_dr

        results.append({
            "Image": fname,
            "fundus_response": fr,
            "dr_response": dr_text,
            "is_fundus": is_fundus,
            "has_dr": has_dr,
            "is_fundus_int": 1 if is_fundus=="Fundus" else 0 if is_fundus=="Not Fundus" else -1,
            "has_dr_int": 1 if has_dr=="DR" else 0 if has_dr=="No DR" else -1,
        })

    df = pd.DataFrame(results).merge(gt, on="Image", how="left")
    try:
        from sklearn.metrics import confusion_matrix, accuracy_score
        import numpy as np
        y_true = df["label"].fillna(-1).astype(int)
        y_pred = df["has_dr_int"]
        m = (y_true!=-1) & (y_pred!=-1)
        yt, yp = y_true[m], y_pred[m]
        if len(yt)>0:
            cm = confusion_matrix(yt, yp, labels=[0,1])
            tn, fp, fn, tp = cm.ravel() if cm.size==4 else (0,0,0,0)
            acc = accuracy_score(yt, yp)
            sn = tp/(tp+fn) if (tp+fn)>0 else 0.0
            sp = tn/(tn+fp) if (tn+fp)>0 else 0.0
            print(f"Accuracy={acc:.4f}  Sensitivity={sn:.4f}  Specificity={sp:.4f}  (TP={tp}, FP={fp}, TN={tn}, FN={fn}, N={len(yt)})")
        else:
            print("No valid labels to compute metrics.")
    except Exception as e:
        print(f"Metrics error: {e}")

    df.to_csv(args.output_csv, index=False)
    print(f"Saved: {args.output_csv}")

if __name__ == "__main__":
    main()
