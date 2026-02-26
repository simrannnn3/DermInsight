import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

# ======================================================
# OPTIONAL PLOTLY
# ======================================================
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

keras.config.enable_unsafe_deserialization()

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config("MSRF-Net | Research XAI", "🧬", layout="wide")

# ======================================================
# CONSTANTS
# ======================================================
MODEL_PATH = "model/best_model.keras"
IMG_SIZE = 256
MC_SAMPLES = 20

# ======================================================
# UTILS
# ======================================================
def normalize(x):
    x = x.astype(np.float32)
    x -= x.min()
    x /= (x.max() + 1e-6)
    return x

# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)

model = load_model()

# ======================================================
# FIND GRAD-CAM LAYER
# ======================================================
def get_last_conv(model):
    for l in reversed(model.layers):
        if isinstance(l, tf.keras.layers.Conv2D) and l.kernel_size != (1,1):
            return l.name
    raise RuntimeError("No Conv layer found")

TARGET_LAYER = get_last_conv(model)

# ======================================================
# PREPROCESS
# ======================================================
def preprocess(img):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    edge = cv2.Canny((arr * 255).astype(np.uint8), 50, 150)
    return (
        np.expand_dims(arr, 0),
        np.expand_dims(edge, (0,3)),
        arr,
        edge
    )

# ======================================================
# GRAD-CAM
# ======================================================
def grad_cam(model, img_t, edge_t, layer):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(layer).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv, pred = grad_model([img_t, edge_t], training=False)
        loss = tf.reduce_mean(pred)
    grads = tape.gradient(loss, conv)
    w = tf.reduce_mean(grads, axis=(1,2), keepdims=True)
    cam = tf.reduce_sum(w * conv, axis=-1)[0].numpy()
    cam = np.maximum(cam, 0)
    return normalize(cv2.resize(cam, (IMG_SIZE, IMG_SIZE)))

# ======================================================
# GRAD-CAM++
# ======================================================
def grad_cam_pp(model, img_t, edge_t, layer):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(layer).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv, pred = grad_model([img_t, edge_t], training=False)
        loss = tf.reduce_mean(pred)
    grads = tape.gradient(loss, conv)
    g2, g3 = grads**2, grads**3
    alpha = g2 / (2*g2 + conv*g3 + 1e-6)
    w = tf.reduce_sum(alpha * tf.nn.relu(grads), axis=(1,2), keepdims=True)
    cam = tf.reduce_sum(w * conv, axis=-1)[0].numpy()
    cam = np.maximum(cam, 0)
    return normalize(cv2.resize(cam, (IMG_SIZE, IMG_SIZE)))

# ======================================================
# ERROR-AWARE XAI
# ======================================================
def error_aware_xai(cam, mask):
    tp = cam * mask
    fp = cam * (1 - mask)
    fn = (1 - cam) * mask

    vis = np.zeros((IMG_SIZE, IMG_SIZE, 3))
    vis[...,1] = tp
    vis[...,0] = fp
    vis[...,2] = fn
    return vis

# ======================================================
# MC DROPOUT
# ======================================================
def mc_dropout(model, img_t, edge_t, n=MC_SAMPLES):
    preds = []
    for _ in range(n):
        p = model([img_t, edge_t], training=True)
        preds.append(p.numpy()[0,:,:,0])
    preds = np.stack(preds)
    mean = preds.mean(0)
    var = preds.var(0)
    return mean, var

# ======================================================
# BOUNDARY UTILS
# ======================================================
def boundary_mask(mask):
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3,3)))

# ======================================================
# HEADER
# ======================================================
st.markdown("## 🧬 MSRF-Net – Skin Lesion Segmentation")
st.caption("Multi-Scale Refinement with Edge-Aware Attention • Research Prototype")
st.warning("⚠️ Research use only. NOT a diagnostic medical tool.")

# ======================================================
# SIDEBAR
# ======================================================
threshold = st.sidebar.slider("Segmentation Threshold", 0.2, 0.8, 0.5, 0.05)

# ======================================================
# INPUT
# ======================================================
up = st.file_uploader("Upload dermoscopic image", ["jpg","png","jpeg"])
if up is None:
    st.stop()

img = Image.open(up)
img_t, edge_t, img_vis, edge_raw = preprocess(img)

# ======================================================
# PREDICTION
# ======================================================
pred = model([img_t, edge_t], training=False).numpy()[0,:,:,0]
mask = (pred >= threshold).astype(np.uint8)

# ======================================================
# VISUAL ANALYSIS
# ======================================================
overlay = img_vis.copy()
overlay[pred > 0.7] = [1,0,0]
overlay[(pred > threshold)&(pred<=0.7)] = [1,1,0]

st.markdown("### 🔬 Visual Analysis")
a,b,c = st.columns(3)
a.image(img_vis, "Original", use_container_width=True)
b.image(overlay, "Confidence Overlay", use_container_width=True)
c.image(edge_raw, "Edge Map", use_container_width=True)

# ======================================================
# PIXEL HOVER
# ======================================================
st.markdown("### 🧪 Pixel-wise Probability Inspection")
if PLOTLY_AVAILABLE:
    fig = px.imshow(pred, color_continuous_scale="Turbo")
    fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# EXPLAINABILITY
# ======================================================
st.markdown("### 🧠 Explainability (Comparative XAI)")
cam_gc = grad_cam(model, img_t, edge_t, TARGET_LAYER)
cam_pp = grad_cam_pp(model, img_t, edge_t, TARGET_LAYER)
err_xai = error_aware_xai(cam_gc, mask)

def overlay_cam(cam):
    heat = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted((img_vis*255).astype(np.uint8),0.6,heat,0.4,0)

x,y,z = st.columns(3)
x.image(overlay_cam(cam_gc), "Grad-CAM", use_container_width=True)
y.image(overlay_cam(cam_pp), "Grad-CAM++", use_container_width=True)
z.image(err_xai, "Error-Aware XAI", use_container_width=True)

# ======================================================
# EPISTEMIC UNCERTAINTY (3 IN A ROW)
# ======================================================
st.markdown("### 🔥 Epistemic Uncertainty (Monte-Carlo Dropout)")
mean_mc, var_mc = mc_dropout(model, img_t, edge_t)

mean_mc_n = normalize(mean_mc)
var_mc_log = normalize(np.log(var_mc + 1e-8))
disagree = normalize(mean_mc_n * var_mc_log)

u1,u2,u3 = st.columns(3)
u1.image(mean_mc_n, "MC Mean Prediction", use_container_width=True)
u2.image(var_mc_log, "MC Variance (log-scaled)", use_container_width=True)
u3.image(disagree, "Confidence–Uncertainty Disagreement", use_container_width=True)

st.metric("Mean Epistemic Uncertainty", f"{var_mc.mean()*1000:.4f}")

# ======================================================
# BOUNDARY ANALYSIS (3 IN A ROW)
# ======================================================
st.markdown("### 🧱 Boundary-Focused Evaluation")
bmask = boundary_mask(mask)
b_unc = normalize(var_mc * bmask)
b_err = np.abs(bmask - (cam_gc > 0.2).astype(np.uint8))

b1,b2,b3 = st.columns(3)
b1.image(bmask*255, "Predicted Boundary", use_container_width=True)
b2.image(b_err*255, "Boundary Error Heatmap", use_container_width=True)
b3.image(b_unc, "Boundary Uncertainty", use_container_width=True)

# ======================================================
# METRICS
# ======================================================
st.markdown("### 📊 Quantitative Metrics")
m1,m2,m3 = st.columns(3)
m1.metric("Mean Confidence", f"{pred.mean()*100:.2f}%")
m2.metric("Lesion Area", f"{mask.mean()*100:.2f}%")
m3.metric("Entropy Uncertainty", f"{np.mean(4*pred*(1-pred))*100:.2f}%")

# ======================================================
# HISTOGRAM
# ======================================================
st.markdown("### 📈 Probability Distribution")
if PLOTLY_AVAILABLE:
    fig = px.histogram(pred.flatten(), nbins=50)
    fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("MSRF-Net • Uncertainty-Aware • Boundary-Aware • Failure-Aware XAI • Academic Use Only")
