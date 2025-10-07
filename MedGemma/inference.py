import os
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
from transformers import AutoProcessor, AutoModelForImageTextToText

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

def generate_response(model, processor, messages, max_new_tokens=100, temperature=0.3):
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
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True
        )
        generated_text = generated_ids[0][input_len:]
        return processor.decode(generated_text, skip_special_tokens=True).strip()

def parse_response_basic(resp):
    low = resp.lower()
    is_fundus = "Fundus" if ("fundus" in low and "not" not in low) else "Not Fundus"
    has_dr = "Unknown"
    if is_fundus == "Fundus":
        if "positive" in low or ("dr" in low and "no" not in low):
            has_dr = "DR"
        elif "negative" in low or "no dr" in low or "no_dr" in low:
            has_dr = "No DR"
    return is_fundus, has_dr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="google/medgemma-27b-it")
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--ground_truth_csv", type=str, required=True)
    ap.add_argument("--output_csv", type=str, required=True)
    ap.add_argument("--apply_clahe", action="store_true")
    ap.add_argument("--clahe_cliplimit", type=float, default=1.0)
    ap.add_argument("--clahe_tilesize", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir,
        device_map="auto", local_files_only=args.local_files_only
    )
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=args.cache_dir, local_files_only=args.local_files_only)

    gt = pd.read_csv(args.ground_truth_csv)
    image_list = gt["Image"].dropna().astype(str).tolist()

    clahe = ApplyCLAHE(args.clahe_cliplimit, args.clahe_tilesize) if args.apply_clahe else None
    rows = []
    for filename in tqdm(image_list, desc="🔍 MedGemma Double Prompt"):
        p = os.path.join(args.image_root, filename)
        if not os.path.exists(p):
            print(f"Missing: {filename}"); continue
        img = Image.open(p).convert("RGB")
        if clahe: img = clahe(img)

        fundus_messages = [
            {"role":"system","content":[{"type":"text","text":"Ophthalmologist: FUNDUS or NOT_FUNDUS only."}]},
            {"role":"user","content":[{"type":"text","text":"Is this fundus?"},{"type":"image","image":img}]}
        ]
        fr = generate_response(model, processor, fundus_messages, args.max_new_tokens, args.temperature)
        is_fundus, has_dr = parse_response_basic(fr); dr_text = "N/A"

        if is_fundus == "Fundus":
            dr_messages = [
                {"role":"system","content":[{"type":"text","text":"Retina specialist: DR_POSITIVE or DR_NEGATIVE only."}]},
                {"role":"user","content":[{"type":"text","text":"DR?"},{"type":"image","image":img}]}
            ]
            dr_text = generate_response(model, processor, dr_messages, args.max_new_tokens, args.temperature)
            low = dr_text.lower()
            if "dr_positive" in low or ("dr" in low and "no" not in low): has_dr = "DR"
            elif "dr_negative" in low or "no dr" in low or "no_dr" in low: has_dr = "No DR"

        rows.append({"Image": filename, "fundus_response": fr, "dr_response": dr_text, "is_fundus": is_fundus, "has_dr": has_dr})

    df = pd.DataFrame(rows).merge(gt, on="Image", how="left")
    from sklearn.metrics import confusion_matrix, accuracy_score
    import numpy as np
    y_true = df.get("label", pd.Series([-1]*len(df))).fillna(-1).astype(int)
    label_map = {"Fundus": 1, "Not Fundus": 0, "DR": 1, "No DR": 0, "Unknown": -1}
    df["has_dr_int"] = df["has_dr"].map(label_map).fillna(-1).astype(int)
    m = (y_true!=-1) & (df["has_dr_int"]!=-1)
    yt, yp = y_true[m], df["has_dr_int"][m]
    if len(yt)>0:
        cm = confusion_matrix(yt, yp, labels=[0,1])
        tn, fp, fn, tp = cm.ravel() if cm.size==4 else (0,0,0,0)
        acc = accuracy_score(yt, yp); sn = tp/(tp+fn) if (tp+fn)>0 else 0.0; sp = tn/(tn+fp) if (tn+fp)>0 else 0.0
        print(f"Accuracy={acc:.4f}  Sensitivity={sn:.4f}  Specificity={sp:.4f}  (TP={tp}, FP={fp}, TN={tn}, FN={fn}, N={len(yt)})")
    else:
        print("No valid labels to compute metrics.")
    df.to_csv(args.output_csv, index=False); print(f"Saved: {args.output_csv}")

if __name__ == "__main__":
    main()
