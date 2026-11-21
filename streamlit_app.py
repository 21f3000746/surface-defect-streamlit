import os
import json
import time
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

st.set_page_config(page_title="SurfaceAI — Surface Defect Segmentation", page_icon="🏭", layout="wide")


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
        return model.eval(), "FPN + efficientnet-b4 (RANDOM weights - model file not found)", 0.3495, 0.1869, None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta", {})
    arch, encoder = meta.get("arch", "fpn"), meta.get("encoder", "efficientnet-b4")
    model = smp.create_model(arch, encoder_name=encoder, encoder_weights=None,
                             in_channels=1, classes=len(CLASS_NAMES))
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict({k.replace("module.", "", 1): v for k, v in state.items()})
    name = f"{arch.upper()} + {encoder}"
    return (model.eval(), name, float(ckpt.get("mean", 0.3495)), float(ckpt.get("std", 0.1869)),
            ckpt.get("val_score"))


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


model, MODEL_NAME, MEAN, STD, VAL_SCORE = load_model()
THRESHOLDS, MIN_AREA, TUNED_WITH_TTA = load_params()


# ------------------------------------------------------------------
# Helpers (same steps as the training notebook)
# ------------------------------------------------------------------
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


# cached: moving the opacity slider re-runs the script but NOT the model
@st.cache_data(max_entries=32, show_spinner=False)
@torch.no_grad()
def predict(gray, use_tta):
    t0 = time.perf_counter()
    small = gray if gray.shape == (IMG_H, IMG_W) else cv2.resize(gray, (IMG_W, IMG_H))
    x = torch.from_numpy((small.astype(np.float32) / 255.0 - MEAN) / STD)[None, None]
    probs = torch.sigmoid(model(x))
    if use_tta:   # original + horizontal + vertical flip, flipped back and averaged
        probs = probs + torch.flip(torch.sigmoid(model(torch.flip(x, dims=[3]))), dims=[3])
        probs = probs + torch.flip(torch.sigmoid(model(torch.flip(x, dims=[2]))), dims=[2])
        probs = probs / 3
    return probs[0].numpy(), (time.perf_counter() - t0) * 1000.0     # (4, 128, 800), ms


def stack_views(views):
    """views = list of (rgb_image, caption, colour). One tall image with a label bar above each strip."""
    rows = []
    for img, caption, color in views:
        w = img.shape[1]
        bar = np.full((28, w, 3), 245, dtype=np.uint8)
        cv2.rectangle(bar, (0, 0), (8, 27), color, -1)
        cv2.putText(bar, caption, (16, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 41, 59), 1, cv2.LINE_AA)
        rows += [bar, img, np.full((6, w, 3), 255, dtype=np.uint8)]
    return np.vstack(rows)


def html(s):
    # strip indentation/blank lines so markdown doesn't turn the HTML into a code block
    st.markdown("\n".join(line.strip() for line in s.splitlines() if line.strip()), unsafe_allow_html=True)


# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------
CSS = """
<style>
.block-container {padding-top: 3.5rem !important; max-width: 1280px;}
.header {padding:18px 22px; border-radius:14px; background:linear-gradient(90deg,#0f172a,#1e3a5f) !important; margin-bottom:12px;}
.header, .header * {color:#ffffff !important;}
.header h1 {margin:0 0 4px 0 !important; padding:0 !important; font-size:1.5rem !important; font-weight:700 !important;}
.header p {margin:0 0 10px 0 !important; color:#cbd5e1 !important; font-size:0.92rem !important;}
.chips {display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px;}
.header .chip {background:rgba(255,255,255,0.14) !important; color:#e2e8f0 !important; padding:3px 10px; border-radius:8px; font-size:0.8rem;}
.header .chip b {color:#ffffff !important;}
.legend {display:flex; gap:8px; flex-wrap:wrap; align-items:center; font-size:0.82rem;}
.header .legend > span:first-child {color:#cbd5e1 !important;}
.header .legend .pill:last-child {color:#111111 !important;}
.pill {padding:3px 12px; border-radius:999px; font-weight:600; font-size:0.8rem;}
.step {font-weight:700; font-size:1.05rem; margin:6px 0 4px 0;}
.status {display:flex; gap:10px; align-items:center; flex-wrap:wrap; padding:12px 16px; border-radius:10px;
         background:#f8fafc !important; border:1px solid #e2e8f0; margin-bottom:10px;}
.status .status-text {font-weight:700;}
.status .status-text.bad {color:#b91c1c !important;}
.status .status-text.good {color:#15803d !important;}
.status .status-meta {margin-left:auto; color:#475569 !important; font-size:0.85rem;}
.placeholder {text-align:center; padding:28px; border:2px dashed #cbd5e1; border-radius:12px; line-height:1.8;}
.placeholder, .placeholder * {color:#64748b !important;}
.cards {display:grid; grid-template-columns:repeat(auto-fit, minmax(250px, 1fr)); gap:12px; margin-top:10px;}
.card {background:#ffffff !important; border:1px solid #e2e8f0; border-radius:12px; padding:12px 14px; box-shadow:0 1px 3px rgba(0,0,0,0.06);}
.card, .card * {color:#0f172a !important;}
.card-head {display:flex; align-items:center; gap:8px; margin-bottom:8px;}
.dot {width:12px; height:12px; border-radius:50%; display:inline-block;}
.badge {margin-left:auto; font-size:0.68rem; font-weight:700; padding:2px 8px; border-radius:6px;}
.card .badge.yes {background:#fee2e2 !important; color:#b91c1c !important;}
.card .badge.no {background:#dcfce7 !important; color:#15803d !important;}
.card-row {display:flex; justify-content:space-between; font-size:0.85rem; padding:2px 0;}
.card .card-row span {color:#475569 !important;}
.bar {height:6px; background:#e2e8f0 !important; border-radius:4px; margin:8px 0 6px 0; overflow:hidden;}
.bar div {height:100%; border-radius:4px;}
.card .card-foot {font-size:0.72rem; color:#94a3b8 !important;}
#MainMenu, footer {visibility:hidden;}
</style>
"""
html(CSS)

pills = "".join(
    f"<span class='pill' style='background:rgb{COLORS[i]}; color:{'#111111' if i == 3 else '#ffffff'} !important;'>{n}</span>"
    for i, n in enumerate(CLASS_NAMES))
score_chip = (f"<span class='chip'>Validation score: <b>{VAL_SCORE:.4f}</b></span>"
              if isinstance(VAL_SCORE, (int, float)) else "")
html(f"""
<div class="header">
  <h1>🏭 SurfaceAI — Industrial Surface Defect Segmentation</h1>
  <p>Finds surface anomalies on steel-strip images, identifies their type and marks the exact pixels they cover.</p>
  <div class="chips">
    <span class="chip">Model: <b>{MODEL_NAME}</b></span>
    <span class="chip">Input: <b>{IMG_W}×{IMG_H} grayscale</b></span>
    <span class="chip">Post-processing: <b>tuned thresholds + min-area filter</b></span>
    {score_chip}
  </div>
  <div class="legend"><span>Legend:</span>{pills}</div>
</div>
""")


# ------------------------------------------------------------------
# Session state: which image is selected (upload or sample)
# ------------------------------------------------------------------
if "sample" not in st.session_state:
    st.session_state.sample = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


def on_upload():
    st.session_state.sample = None          # a new upload replaces the selected sample


def pick_sample(path):
    st.session_state.sample = path
    st.session_state.uploader_key += 1      # clears the uploader so the sample is shown


def clear_all():
    st.session_state.sample = None
    st.session_state.uploader_key += 1


left, right = st.columns([2, 3], gap="large")

# ---------------- left: input + settings ----------------
with left:
    html("<div class='step'>① Choose an image</div>")
    upload = st.file_uploader("Upload a steel-strip image (jpg / png)", type=["jpg", "jpeg", "png"],
                              key=f"uploader_{st.session_state.uploader_key}", on_change=on_upload)

    examples = sorted(f for f in os.listdir("examples")
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))) if os.path.isdir("examples") else []
    if examples:
        st.caption("…or click a sample strip to inspect it")
        cols = st.columns(2)
        for i, f in enumerate(examples):
            path = os.path.join("examples", f)
            with cols[i % 2]:
                st.image(path, width="stretch")
                st.button("Inspect" if st.session_state.sample != path else "✓ Selected",
                          key=f"ex_{i}", on_click=pick_sample, args=(path,), width="stretch",
                          type="primary" if st.session_state.sample == path else "secondary")

    html("<div class='step'>② Settings</div>")
    use_tta = st.checkbox("3-view flip TTA (more accurate, ~3× slower)", value=TUNED_WITH_TTA)
    alpha = st.slider("Mask opacity", 0.1, 0.9, 0.5, 0.1)
    st.button("Clear", on_click=clear_all, width="stretch")

gray, source = None, ""
if upload is not None:
    gray, source = read_upload(upload), upload.name
elif st.session_state.sample:
    gray = cv2.imread(st.session_state.sample, cv2.IMREAD_GRAYSCALE)
    source = os.path.basename(st.session_state.sample)

# ---------------- right: results ----------------
with right:
    html("<div class='step'>③ Results</div>")

    if gray is None:
        html("""
        <div class="placeholder">
          <div style="font-size:2rem;">🔍</div>
          <div><b>No inspection yet</b></div>
          <div>Upload a steel-strip image or click one of the samples — inspection starts automatically.</div>
        </div>""")
        st.stop()

    with st.spinner("Running the model…"):
        probs, ms = predict(gray, use_tta)

    orig_h, orig_w = gray.shape
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    display = gray_rgb.astype(np.float32)
    rows, mask_views, heat_views = [], [], []
    for c, name in enumerate(CLASS_NAMES):
        mask = remove_small_components((probs[c] > THRESHOLDS[c]).astype(np.uint8), MIN_AREA[c])
        n_pixels = int(mask.sum())
        area = 100 * n_pixels / mask.size
        conf = float(probs[c].max())
        rows.append((name, n_pixels > 0, n_pixels, area, conf))

        show = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        if n_pixels:
            display[show == 1] = (1 - alpha) * display[show == 1] + alpha * np.array(COLORS[c], dtype=np.float32)
            contours, _ = cv2.findContours(show, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, COLORS[c], 2)

        single = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        single[show == 1] = COLORS[c]
        mask_views.append((single, f"{name}: {'DETECTED' if n_pixels else 'empty'}  ({area:.2f}% of image)", COLORS[c]))

        prob = cv2.resize(probs[c], (orig_w, orig_h))
        heat = cv2.cvtColor(cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        heat_views.append((cv2.addWeighted(gray_rgb, 0.4, heat, 0.6, 0),
                           f"{name}: peak confidence {conf:.2f}", COLORS[c]))

    # status banner
    found = [r[0] for r in rows if r[1]]
    if found:
        color, icon, cls = "#b91c1c", "⚠️", "bad"
        text = f"{len(found)} defect type{'s' if len(found) > 1 else ''} detected: " + ", ".join(found)
    else:
        color, icon, cls = "#15803d", "✅", "good"
        text = "Clean strip — no defects detected"
    extra = f" · resized {orig_w}×{orig_h} → {IMG_W}×{IMG_H}" if (orig_h, orig_w) != (IMG_H, IMG_W) else ""
    html(f"""
    <div class="status" style="border-left:6px solid {color};">
      <span style="font-size:1.3rem;">{icon}</span>
      <span class="status-text {cls}">{text}</span>
      <span class="status-meta">⏱ {ms:.0f} ms on CPU{extra}</span>
    </div>""")

    tab1, tab2, tab3 = st.tabs(["Overlay", "Per-class masks", "Confidence heatmaps"])
    with tab1:
        st.image(display.clip(0, 255).astype(np.uint8), caption=f"Predicted defect overlay — {source}",
                 width="stretch")
    with tab2:
        st.image(stack_views(mask_views), width="stretch")
    with tab3:
        st.image(stack_views(heat_views), width="stretch")
        st.caption("Colour scale: **blue = low** → **red = high** defect probability.")

    # result cards
    cards = ""
    for i, (name, present, n_pixels, area, conf) in enumerate(rows):
        r, g, b = COLORS[i]
        badge = "<span class='badge yes'>DETECTED</span>" if present else "<span class='badge no'>NOT PRESENT</span>"
        cards += f"""
        <div class="card" style="border-top:5px solid rgb({r},{g},{b}); opacity:{1 if present else 0.75};">
          <div class="card-head"><span class="dot" style="background:rgb({r},{g},{b});"></span><b>{name}</b>{badge}</div>
          <div class="card-row"><span>Area covered</span><b>{area:.2f}%</b></div>
          <div class="card-row"><span>Pixels</span><b>{n_pixels:,}</b></div>
          <div class="card-row"><span>Peak confidence</span><b>{conf:.2f}</b></div>
          <div class="bar"><div style="width:{conf * 100:.0f}%; background:rgb({r},{g},{b});"></div></div>
          <div class="card-foot">thr {THRESHOLDS[i]:.2f} · min area {MIN_AREA[i]}px</div>
        </div>"""
    html(f"<div class='cards'>{cards}</div>")

# ---------------- bottom: details ----------------
with st.expander("📋 Detailed results table"):
    st.dataframe(pd.DataFrame(
        [{"Defect type": r[0], "Present": "Yes" if r[1] else "No", "Pixels": r[2],
          "Area %": round(r[3], 3), "Max confidence": round(r[4], 3)} for r in rows]),
        hide_index=True, width="stretch")

with st.expander("ℹ️ About the model"):
    st.markdown(f"""
**How it works**
1. The image is read as grayscale and resized to {IMG_W}×{IMG_H} (the training resolution), then normalised.
2. **{MODEL_NAME}** predicts one probability map per defect type (sigmoid, multi-label — a strip can have several defect types).
3. Optional **3-view flip TTA** averages predictions over the original, horizontal flip and vertical flip.
4. Each map is binarised with a **per-class tuned threshold**, and connected regions smaller than the
   **per-class minimum area** are removed to avoid false positives.
""")
    st.table(pd.DataFrame({"Class": CLASS_NAMES, "Threshold": [f"{t:.2f}" for t in THRESHOLDS],
                           "Min area (px)": MIN_AREA}).set_index("Class"))
    st.markdown("**Competition metric:** 0.70 × Foreground Dice + 0.20 × False-Positive score + 0.10 × Empty accuracy.")
