import os
import json
import numpy as np
import pandas as pd
import torch
import cv2
import streamlit as st
import segmentation_models_pytorch as smp

# ------------------------------------------------------------------
# SETTINGS
# The checkpoint holds the architecture, encoder, mean and std; postprocess_params.json
# (same Kaggle Model folder) holds the tuned thresholds and minimum areas.
# ------------------------------------------------------------------
MODEL_PATH = "m9_fpn_effb4.pth"
PARAMS_PATH = "postprocess_params.json"
# if the files are not in this repo (GitHub web upload stops at 25 MB), they are downloaded
# from this public Hugging Face repo, which already holds them
HF_REPO, HF_REPO_TYPE = "21f3000746/SurfaceAI", "space"
IMG_H, IMG_W = 128, 800
CLASS_NAMES = ["defect_a", "defect_b", "defect_c", "defect_d"]
DEFAULT_THRESHOLDS = [0.5, 0.5, 0.5, 0.5]
DEFAULT_MIN_AREA = [0, 0, 0, 0]
COLORS = [(255, 60, 60), (60, 200, 60), (60, 120, 255), (255, 200, 0)]   # RGB per class

st.set_page_config(page_title="Surface Defect Segmentation", page_icon="🔍", layout="wide")


# ------------------------------------------------------------------
# Load model + post-processing parameters once (cached across reruns)
# ------------------------------------------------------------------
def fetch(filename):
    if os.path.exists(filename):
        return filename
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=HF_REPO, repo_type=HF_REPO_TYPE, filename=filename)
    except Exception as e:
        print(f"could not get {filename}: {e}")
        return None


@st.cache_resource
def load_model():
    path = fetch(MODEL_PATH)
    if path is None:
        model = smp.create_model("fpn", encoder_name="efficientnet-b4", encoder_weights=None,
                                 in_channels=1, classes=len(CLASS_NAMES))
        return model.eval(), "FPN + efficientnet-b4 (RANDOM weights - model file not found)", 0.3495, 0.1869
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta", {})
    arch, encoder = meta.get("arch", "fpn"), meta.get("encoder", "efficientnet-b4")
    model = smp.create_model(arch, encoder_name=encoder, encoder_weights=None,
                             in_channels=1, classes=len(CLASS_NAMES))
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict({k.replace("module.", "", 1): v for k, v in state.items()})
    name = f"{arch.upper()} + {encoder} (epoch {ckpt.get('epoch')})"
    return model.eval(), name, float(ckpt.get("mean", 0.3495)), float(ckpt.get("std", 0.1869))


@st.cache_data
def load_params():
    path = fetch(PARAMS_PATH)
    if path is None:
        return DEFAULT_THRESHOLDS, DEFAULT_MIN_AREA, False
    with open(path) as f:
        p = json.load(f)
    return ([float(p["thresholds"][c]) for c in CLASS_NAMES],
            [int(p["min_areas"][c]) for c in CLASS_NAMES],
            "hflip" in p.get("tta", []))


model, MODEL_NAME, MEAN, STD = load_model()
THRESHOLDS, MIN_AREA, TUNED_WITH_TTA = load_params()


# ------------------------------------------------------------------
# Helpers (same steps as the training notebook)
# ------------------------------------------------------------------
def mask_to_rle(mask):
    pixels = np.concatenate([[0], mask.flatten(order="F"), [0]])     # column-major, 1-indexed
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def remove_small_components(mask, min_area):
    if min_area <= 0 or mask.sum() == 0:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels].astype(np.uint8)


def read_upload(file):
    data = np.frombuffer(file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)       # same reader as in training


@torch.no_grad()
def predict(gray, use_tta):
    small = gray if gray.shape == (IMG_H, IMG_W) else cv2.resize(gray, (IMG_W, IMG_H))
    x = torch.from_numpy((small.astype(np.float32) / 255.0 - MEAN) / STD)[None, None]
    probs = torch.sigmoid(model(x))
    if use_tta:   # original + horizontal + vertical flip, flipped back and averaged
        probs = probs + torch.flip(torch.sigmoid(model(torch.flip(x, dims=[3]))), dims=[3])
        probs = probs + torch.flip(torch.sigmoid(model(torch.flip(x, dims=[2]))), dims=[2])
        probs = probs / 3
    return probs[0].numpy()                               # (4, 128, 800)


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("🔍 Industrial Surface Defect Segmentation")
st.write(f"Upload a 128×800 grayscale strip image. Model: **{MODEL_NAME}**. "
         "It finds surface anomalies, identifies their type and marks the pixels they cover.")
st.markdown(" | ".join(f"<span style='color:rgb{COLORS[i]}'>■</span> {n}"
                       for i, n in enumerate(CLASS_NAMES)), unsafe_allow_html=True)

with st.sidebar:
    st.header("Input")
    upload = st.file_uploader("Image (jpg / png)", type=["jpg", "jpeg", "png"])
    examples = sorted(f for f in os.listdir("examples")
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))) if os.path.isdir("examples") else []
    example = st.selectbox("…or pick an example", ["(none)"] + examples) if examples else "(none)"
    use_tta = st.checkbox("Flip TTA" + (" (thresholds were tuned with it on)" if TUNED_WITH_TTA else ""),
                          value=TUNED_WITH_TTA)
    st.caption(f"Thresholds: {THRESHOLDS} · min areas: {MIN_AREA}")

gray = None
if upload is not None:
    gray = read_upload(upload)
elif example != "(none)":
    gray = cv2.imread(os.path.join("examples", example), cv2.IMREAD_GRAYSCALE)

if gray is None:
    st.info("Upload an image or pick an example in the sidebar.")
    st.stop()

with st.spinner("Running the model…"):
    probs = predict(gray, use_tta)

orig_h, orig_w = gray.shape
display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
rows, rle_rows = [], []
for c, name in enumerate(CLASS_NAMES):
    mask = remove_small_components((probs[c] > THRESHOLDS[c]).astype(np.uint8), MIN_AREA[c])
    n_pixels = int(mask.sum())
    rows.append({"Defect type": name, "Present": "Yes" if n_pixels else "No", "Pixels": n_pixels,
                 "Area %": round(100 * n_pixels / mask.size, 3), "Max confidence": round(float(probs[c].max()), 3)})
    rle_rows.append({"defect_type": name, "mask_rle": mask_to_rle(mask) if n_pixels else " "})
    if n_pixels:
        show = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        display[show == 1] = 0.5 * display[show == 1] + 0.5 * np.array(COLORS[c], dtype=np.float32)
        contours, _ = cv2.findContours(show, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display, contours, -1, COLORS[c], 2)

found = [r["Defect type"] for r in rows if r["Present"] == "Yes"]
if found:
    st.warning("Defects found: " + ", ".join(found))
else:
    st.success("No defects found")
if (orig_h, orig_w) != (IMG_H, IMG_W):
    st.caption(f"Image resized from {orig_w}×{orig_h} to {IMG_W}×{IMG_H}; RLE refers to {IMG_W}×{IMG_H}.")

st.image(gray, caption="Input", width="stretch")
st.image(display.clip(0, 255).astype(np.uint8), caption="Predicted defects", width="stretch")
st.subheader("Per-class results")
st.dataframe(pd.DataFrame(rows), hide_index=True)

st.subheader("RLE output (competition format, column-major)")
rle_df = pd.DataFrame(rle_rows)
st.code(rle_df.to_csv(index=False), language="text")
st.download_button("Download masks as CSV", rle_df.to_csv(index=False), "masks.csv", "text/csv")
