# app.py
import io
import os
import time
import base64
import textwrap
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# ---------- Optional deps with graceful fallback ----------
# OpenCV for heatmap overlay
try:
    import cv2
    _HAS_CV2 = True
except Exception as _e_cv2:
    _HAS_CV2 = False
    _CV2_ERR = str(_e_cv2)

# ReportLab for PDF report
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    _HAS_REPORTLAB = True
except Exception as _e_rl:
    _HAS_REPORTLAB = False
    _RL_ERR = str(_e_rl)

# ---------- SINGLE page_config (removed duplicate) ----------
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide"
)


# -------------- Globals --------------
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
TARGET_SIZE = (224, 224)
MODEL_PATH = 'brain_tumor_model.h5'  # <-- ensure this file is present
GRADCAM_SKIP_CONFIDENCE = 99.0       # skip Grad-CAM if prediction is super confident
TUMOR_DECISION_DEFAULT = 0.50        # >>> default threshold for tumor vs no tumor

# -------------- Utils --------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return tf.keras.models.load_model(path)

def preprocess_image(pil_image: Image.Image, target_size=TARGET_SIZE):
    img = pil_image.convert("RGB").resize(target_size)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0), img  # model_input, resized_rgb

def find_last_conv_layer(model: tf.keras.Model):
    """
    Recursively find the last Conv2D layer object in the model.
    Works for nested Sequential or Functional models.
    Returns the actual layer object or None.
    """
    last_conv = None

    def _search_layers(layers):
        nonlocal last_conv
        for layer in layers:
            # direct Conv2D
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer
            # nested Model/Sequential: search inside
            elif isinstance(layer, (tf.keras.Model, tf.keras.Sequential)):
                try:
                    _search_layers(layer.layers)
                except Exception:
                    children = getattr(layer, "layers", None)
                    if children:
                        _search_layers(children)

    top_layers = getattr(model, "layers", None)
    if top_layers:
        _search_layers(top_layers)
    return last_conv

def find_layer_by_name(model: tf.keras.Model, name: str):
    """
    Recursively find a layer object within `model` whose .name equals `name`.
    Returns the layer object or None.
    """
    if not name:
        return None
    try:
        return model.get_layer(name)
    except Exception:
        pass

    found = None

    def _search(layers):
        nonlocal found
        for layer in layers:
            if layer.name == name:
                found = layer
                return True
            if isinstance(layer, (tf.keras.Model, tf.keras.Sequential)):
                children = getattr(layer, "layers", None)
                if children:
                    if _search(children):
                        return True
        return False

    top_layers = getattr(model, "layers", None)
    if top_layers:
        _search(top_layers)
    return found

def _resolve_conv_layer_from_name_or_auto(name: str | None, model: tf.keras.Model, auto_layer: tf.keras.layers.Layer | None):
    """
    Resolve a conv layer object from a user-provided name (string) or fall back to auto_layer.
    This helper is kept for compatibility but the single-prediction flow uses auto-detected layer object (see below).
    """
    if (not name or not isinstance(name, str) or not name.strip()) and auto_layer is not None:
        resolved = find_layer_by_name(model, getattr(auto_layer, "name", "")) or auto_layer
        return resolved

    if isinstance(name, str) and name.strip():
        lname = name.strip()
        try:
            layer = model.get_layer(lname)
            if isinstance(layer, (tf.keras.Model, tf.keras.Sequential)):
                inner_conv = find_last_conv_layer(layer)
                if inner_conv is not None:
                    resolved = find_layer_by_name(model, inner_conv.name)
                    return resolved if resolved is not None else inner_conv
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer
            return layer
        except Exception:
            pass

        def _search_conv_by_substr(layers):
            for l in layers:
                if isinstance(l, tf.keras.layers.Conv2D) and (lname in l.name or l.name in lname or l.name.endswith(lname)):
                    return l
                if isinstance(l, (tf.keras.Model, tf.keras.Sequential)):
                    children = getattr(l, "layers", None)
                    if children:
                        found = _search_conv_by_substr(children)
                        if found:
                            return found
            return None

        top_layers = getattr(model, "layers", None)
        if top_layers:
            found_conv = _search_conv_by_substr(top_layers)
            if found_conv is not None:
                resolved = find_layer_by_name(model, found_conv.name)
                return resolved if resolved is not None else found_conv

    if auto_layer is not None:
        resolved_auto = find_layer_by_name(model, getattr(auto_layer, "name", "")) or auto_layer
        return resolved_auto

    return None

def make_gradcam_heatmap(model: tf.keras.Model, img_tensor: np.ndarray, conv_layer):
    """
    Returns a normalized heatmap (H, W) from Grad-CAM for the predicted class.
    Accepts:
      - conv_layer: a Conv2D layer object (preferred) or a string layer name (will be resolved).
    Guarantees model is called once on the input so layers have defined outputs.
    Includes a robust fallback when the CAM is all zeros.
    """
    # Resolve layer if name provided
    if isinstance(conv_layer, str):
        conv_layer_obj = find_layer_by_name(model, conv_layer)
    else:
        conv_layer_obj = conv_layer

    # If a nested model/seq was provided, find its last Conv2D and resolve to top-level instance
    if isinstance(conv_layer_obj, (tf.keras.Model, tf.keras.Sequential)):
        inner = find_last_conv_layer(conv_layer_obj)
        conv_layer_obj = find_layer_by_name(model, inner.name) if (inner is not None) else None

    if conv_layer_obj is None or not isinstance(conv_layer_obj, tf.keras.layers.Conv2D):
        # try heuristic by name attribute
        candidate_name = getattr(conv_layer_obj, "name", None) if conv_layer_obj is not None else None
        if candidate_name:
            conv_layer_obj = find_layer_by_name(model, candidate_name)

    if conv_layer_obj is None or not hasattr(conv_layer_obj, "output"):
        raise ValueError("Could not resolve a usable Conv2D layer for Grad-CAM (ensure model contains Conv2D layers).")

    # Ensure model is called once on the input so the internal graph tensors exist
    # This is critical to avoid "layer has never been called" errors
    try:
        _ = model(img_tensor, training=False)
    except Exception:
        pass  # ignore if calling again is unnecessary

    # Build grad model from inputs -> [target_layer.output, model.output]
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[conv_layer_obj.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_tensor)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    conv_outputs = conv_outputs[0]        # (H, W, C)
    grads = grads[0]                      # (H, W, C)

    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1).numpy()

    cam = np.maximum(cam, 0)

    if cam.max() == 0:
        try:
            conv_np = conv_outputs.numpy() if hasattr(conv_outputs, "numpy") else np.array(conv_outputs)
            fallback = np.abs(np.mean(conv_np, axis=-1))
            if fallback.max() != 0:
                cam = fallback / fallback.max()
            else:
                cam = np.ones_like(cam) * 1e-6
        except Exception:
            cam = np.ones_like(cam) * 1e-6
    else:
        cam = cam / cam.max()

    return cam

def _overlay_with_cv2(heatmap: np.ndarray, base_img_rgb: Image.Image, alpha=0.4):
    hmap = cv2.resize(heatmap, (base_img_rgb.width, base_img_rgb.height))
    hmap = np.uint8(255 * hmap)
    hmap_color = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)

    base_bgr = cv2.cvtColor(np.array(base_img_rgb), cv2.COLOR_RGB2BGR)
    superimposed = cv2.addWeighted(base_bgr, 1 - alpha, hmap_color, alpha, 0)
    super_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
    return Image.fromarray(super_rgb)

def _overlay_with_pil(heatmap: np.ndarray, base_img_rgb: Image.Image, alpha=0.4):
    hm = (heatmap * 255).astype(np.uint8)
    hm = Image.fromarray(hm).resize(base_img_rgb.size, Image.BILINEAR).convert("L")
    color_overlay = Image.new("RGBA", base_img_rgb.size, (255, 0, 0, 0))
    color_overlay.putalpha(hm)
    blended = Image.alpha_composite(base_img_rgb.convert("RGBA"), color_overlay)
    return blended.convert("RGB")

def overlay_heatmap_on_image(heatmap: np.ndarray, base_img_rgb: Image.Image, alpha=0.4):
    if _HAS_CV2:
        return _overlay_with_cv2(heatmap, base_img_rgb, alpha=alpha)
    else:
        return _overlay_with_pil(heatmap, base_img_rgb, alpha=alpha)

def predict_single(model, pil_image: Image.Image):
    tensor, resized = preprocess_image(pil_image)
    preds = model.predict(tensor, verbose=0)[0]
    return preds, resized, tensor

def binary_tumor_decision(probs, class_names=CLASS_NAMES, threshold=TUMOR_DECISION_DEFAULT):
    try:
        idx_no = class_names.index("No Tumor")
    except ValueError:
        return None, None
    no_tumor_prob = float(probs[idx_no])
    tumor_prob = 1.0 - no_tumor_prob
    binary_label = "Tumor Detected" if tumor_prob >= float(threshold) else "No Tumor"
    return binary_label, tumor_prob

def build_pdf_report(
    patient_id: str,
    original_img: Image.Image,
    heatmap_img: Image.Image,
    predicted_class: str,
    confidence: float,
    probs: np.ndarray,
    class_names=CLASS_NAMES,
    # >>> optional binary fields
    binary_label: str | None = None,
    tumor_prob: float | None = None,
):
    """
    Returns a BytesIO object containing the PDF bytes.
    """
    if not _HAS_REPORTLAB:
        raise RuntimeError(f"ReportLab is not installed: {_RL_ERR}")

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    margin = 40
    y = height - margin

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Brain Tumor Detection Report")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 14
    if patient_id:
        c.drawString(margin, y, f"Patient ID: {patient_id}")
        y -= 14

    # Prediction summary
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Prediction Summary")
    y -= 16
    c.setFont("Helvetica", 11)
    # >>> Binary line first
    if binary_label is not None and tumor_prob is not None:
        c.drawString(margin, y, f"Binary Decision: {binary_label} (Tumor prob: {tumor_prob*100:.2f}%)")
        y -= 14
    c.drawString(margin, y, f"Predicted Class: {predicted_class}")
    y -= 14
    c.drawString(margin, y, f"Confidence: {confidence:.2f}%")
    y -= 16

    # >>> Tumor status line (colored)
    try:
        if isinstance(predicted_class, str) and predicted_class.lower() == "no tumor":
            c.setFillColorRGB(0, 0.5, 0)  # green
            c.setFont("Helvetica-Bold", 11)
            c.drawString(margin, y, f"Result: No tumor detected with {confidence:.2f}% confidence.")
            y -= 16
        else:
            c.setFillColorRGB(1, 0, 0)  # red
            c.setFont("Helvetica-Bold", 11)
            c.drawString(margin, y, f"Result: Tumor confirmed ({predicted_class}) with {confidence:.2f}% confidence.")
            y -= 16
    except Exception:
        # if something goes wrong, continue without stopping PDF generation
        pass
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 10)

    # Probabilities table
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Class Probabilities:")
    y -= 14
    c.setFont("Helvetica", 10)
    for cls, p in zip(class_names, probs):
        c.drawString(margin + 12, y, f"- {cls}: {p*100:.2f}%")
        y -= 12

    y -= 10
    # Images: original and heatmap
    max_img_width = (width - 3 * margin) / 2
    # Convert PIL to ImageReader
    orig_reader = ImageReader(original_img)
    heat_reader = ImageReader(heatmap_img)

    img_h = max_img_width  # square slot
    c.drawImage(orig_reader, margin, y - img_h, width=max_img_width, height=img_h, preserveAspectRatio=True, mask='auto')
    c.drawImage(heat_reader, margin + max_img_width + margin, y - img_h, width=max_img_width, height=img_h,
                preserveAspectRatio=True, mask='auto')
    y -= img_h + 10

    # Footer / Disclaimer
    disclaimer = (
        "Disclaimer: This report is generated by a machine learning model for educational and research purposes. "
        "It is NOT a substitute for professional medical advice, diagnosis, or treatment."
    )
    wrapped = textwrap.fill(disclaimer, 95)
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(margin, y, "Disclaimer:")
    y -= 12
    for line in wrapped.split('\n'):
        c.drawString(margin, y, line)
        y -= 12

    # Developer credit in footer
    try:
        c.setFont("Helvetica-Bold", 9)
        c.drawString(margin, y - 10, "Report generated by Brain Tumor Detection App (Developed by ARMAN)")
        y -= 14
    except Exception:
        pass

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# -------------- Sidebar --------------
st.sidebar.title("🧠 Brain Tumor MRI Classifier")
# show developer credit in the sidebar per your request
st.sidebar.markdown("**Developed by ARMAN**")
st.sidebar.info("Classes: " + ", ".join(CLASS_NAMES))
st.sidebar.markdown("---")
st.sidebar.caption("Tip: For the dashboard, upload a CSV with columns: `image, true_label, pred_label` (optional).")

# -------------- Model Load (with file check) --------------
load_error = None
model = None
if not os.path.exists(MODEL_PATH):
    load_error = f"Model file not found at: {MODEL_PATH}"
else:
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        load_error = str(e)

if not _HAS_CV2:
    st.sidebar.warning(f"OpenCV (cv2) not found. Heatmap overlay will use a fallback method. ({_CV2_ERR})")

if not _HAS_REPORTLAB:
    st.sidebar.warning(f"ReportLab not found. PDF report generation will be disabled. ({_RL_ERR})")

# Add a prominent front title on the app main page
st.title("🧠 Brain Tumor Detection App")

tabs = st.tabs(["🔎 Predict (Single)", "📦 Batch Predict", "📊 Dashboard", "ℹ️ About"])

# -------------- TAB 1: Single Prediction --------------
with tabs[0]:
    st.header("Single MRI Prediction with Grad-CAM")
    # >>> developer caption under header
    st.caption("👨‍💻 Developed by ARMAN")
    if load_error:
        st.error(f"Model failed to load: {load_error}")
    else:
        colA, colB = st.columns([1.1, 1])
        with colA:
            patient_id = st.text_input("Patient ID (optional)", value="")
            uploaded_file = st.file_uploader("Upload an MRI image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
            gen_report = st.checkbox("Generate PDF report after prediction", value=True if _HAS_REPORTLAB else False)
            if gen_report and not _HAS_REPORTLAB:
                st.info("Install `reportlab` to enable PDF report: `pip install reportlab`")

        with colB:
            st.write("**Prediction Settings**")
            # auto-detect layer object, and show its name in text input (info only)
            auto_last_conv_layer = find_last_conv_layer(model)
            auto_last_conv_name = auto_last_conv_layer.name if (auto_last_conv_layer is not None and hasattr(auto_last_conv_layer, "name")) else ""
            layer_name = st.text_input("Grad-CAM last conv layer name", value=(auto_last_conv_name or ""))
            alpha = st.slider("Heatmap overlay alpha", 0.0, 1.0, 0.40, 0.05)
            # >>> Add tumor decision threshold control
            tumor_thresh = st.slider("Tumor decision threshold", 0.10, 0.90, TUMOR_DECISION_DEFAULT, 0.05)
            if not layer_name:
                st.warning("Could not auto-detect a Conv2D/feature map layer. Enter one manually if your model has it.")

        if uploaded_file:
            try:
                # Load original image (keep ORIGINAL for overlay)
                raw_img = Image.open(uploaded_file).convert("RGB")

                c1, c2 = st.columns(2)
                with c1:
                    st.image(raw_img, caption="Uploaded MRI", use_container_width=True)

                with st.spinner("Analyzing image..."):
                    probs, resized_rgb, tensor = predict_single(model, raw_img)
                    pred_idx = int(np.argmax(probs))
                    pred_class = CLASS_NAMES[pred_idx]
                    confidence = float(np.max(probs) * 100)

                    # >>> Binary decision
                    bin_label, tumor_prob = binary_tumor_decision(probs, CLASS_NAMES, tumor_thresh)

                    # Grad-CAM (always attempt when tumor detected and a conv layer is available; ignore confidence)
                    overlay = None
                    if bin_label == "Tumor Detected":
                        try:
                            # Use the auto-detected Conv2D layer object (do NOT use user-supplied 'sequential' string)
                            conv_layer_obj = auto_last_conv_layer
                            if conv_layer_obj is None:
                                st.info("Grad-CAM not generated (no conv layer found).")
                            else:
                                heatmap = make_gradcam_heatmap(model, tensor, conv_layer_obj)
                                # Make overlay highly visible: ensure alpha at least 0.75
                                overlay = overlay_heatmap_on_image(heatmap, raw_img, alpha=max(alpha, 0.75))
                        except Exception as e_cam:
                            st.warning(f"Grad-CAM failed: {e_cam}")
                    elif bin_label != "Tumor Detected":
                        st.info("Grad-CAM not generated for 'No Tumor' decision.")
                    elif not layer_name:
                        st.info("Grad-CAM not generated (no conv layer name provided).")

                with c2:
                    # >>> Show binary result prominently (mapping internal label to wording)
                    if bin_label == "Tumor Detected":
                        # Show tumor probability (1 - P(no tumor))
                        st.error(f"⚠️ Tumor confirmed ({pred_class}) — probability {tumor_prob*100:.2f}%")
                    elif bin_label == "No Tumor":
                        # Show no-tumor probability = P(no tumor) = 1 - tumor_prob
                        st.success(f"✅ No tumor detected — probability {(1 - tumor_prob)*100:.2f}%")
                    else:
                        st.info("Binary decision unavailable (couldn't locate 'No Tumor' class).")

                    if overlay is not None:
                        st.image(overlay, caption="Grad-CAM Heatmap", use_container_width=True)

                    st.markdown(f"### 🧾 Predicted subtype: **{pred_class}**")
                    st.markdown(f"**Subtype confidence:** `{confidence:.2f}%`")

                    prob_df = pd.DataFrame({
                        "Tumor Type": CLASS_NAMES,
                        "Probability (%)": probs * 100.0
                    }).set_index("Tumor Type")
                    st.bar_chart(prob_df)

                # PDF download
                if gen_report:
                    try:
                        pdf_buf = build_pdf_report(
                            patient_id=patient_id.strip(),
                            original_img=raw_img,      # use original image
                            heatmap_img=(overlay if overlay is not None else resized_rgb),
                            predicted_class=pred_class,
                            confidence=confidence,
                            probs=probs,
                            # >>> pass binary info
                            binary_label=bin_label,
                            tumor_prob=tumor_prob
                        )
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_buf,
                            file_name=f"Brain_Tumor_Report_{patient_id or 'patient'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e_pdf:
                        st.error(f"Failed to build PDF: {e_pdf}")

            except Exception as e:
                st.error(f"Error processing the image: {e}")

# -------------- TAB 2: Batch Prediction --------------
# -------------- TAB 2: Batch Prediction --------------
with tabs[1]:
    st.header("Batch Predictions (Multiple Images)")
    if load_error:
        st.error(f"Model failed to load: {load_error}")
    else:
        batch_files = st.file_uploader(
            "Upload multiple MRI images", 
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        show_heatmaps = st.checkbox("Generate & display heatmaps for each image (slower)", value=False)
        alpha_b = st.slider("Heatmap overlay alpha (batch)", 0.0, 1.0, 0.40, 0.05)
        tumor_thresh_batch = st.slider("Tumor decision threshold (batch)", 0.10, 0.90, TUMOR_DECISION_DEFAULT, 0.05)

        if batch_files:
            results = []
            gallery = []
            auto_last_conv_b = find_last_conv_layer(model)

            with st.spinner("Running predictions..."):
                for file in batch_files:
                    try:
                        raw = Image.open(file).convert("RGB")
                        probs, resized_rgb, tensor = predict_single(model, raw)
                        pred_idx = int(np.argmax(probs))
                        pred_class = CLASS_NAMES[pred_idx]
                        conf = float(np.max(probs) * 100)

                        # Binary decision
                        bin_label_b, tumor_prob_b = binary_tumor_decision(probs, CLASS_NAMES, tumor_thresh_batch)

                        # Extract true_label from filename using better pattern matching
                        true_label = ""
                        filename = file.name.lower()
                        
                        # Check for each class name in filename
                        for class_name in CLASS_NAMES:
                            class_lower = class_name.lower()
                            # Look for class name in filename (e.g., "glioma", "meningioma", etc.)
                            if class_lower in filename:
                                true_label = class_name
                                break
                        
                        # If not found, try more specific patterns
                        if not true_label:
                            if "no" in filename and "tumor" in filename:
                                true_label = "No Tumor"
                            elif "gl" in filename:
                                true_label = "Glioma"
                            elif "me" in filename:
                                true_label = "Meningioma"
                            elif "pi" in filename or "pt" in filename:
                                true_label = "Pituitary"

                        row = {
                            "image": file.name,
                            "true_label": true_label,
                            "pred_label": pred_class,
                            "confidence_%": round(conf, 2),
                            "tumor_detected": (bin_label_b == "Tumor Detected"),
                            "tumor_prob_%": round((tumor_prob_b or 0.0) * 100, 2),
                            **{f"prob_{c}": float(p*100) for c, p in zip(CLASS_NAMES, probs)}
                        }
                        results.append(row)

                        # Heatmap
                        if show_heatmaps and bin_label_b == "Tumor Detected" and auto_last_conv_b:
                            try:
                                hmap = make_gradcam_heatmap(model, tensor, auto_last_conv_b)
                                overlay = overlay_heatmap_on_image(hmap, raw, alpha=max(alpha_b, 0.75))
                                gallery.append((file.name, raw, overlay))
                            except Exception as e_cam_b:
                                gallery.append((file.name, raw, f"Grad-CAM error: {e_cam_b}"))
                        elif show_heatmaps:
                            gallery.append((file.name, raw, "No heatmap for 'No Tumor' or conv layer missing"))

                    except Exception as e:
                        results.append({
                            "image": file.name, 
                            "true_label": "", 
                            "pred_label": f"ERROR: {e}", 
                            "confidence_%": 0.0,
                            "tumor_detected": False,
                            "tumor_prob_%": 0.0,
                            **{f"prob_{c}": 0.0 for c in CLASS_NAMES}
                        })

            df_results = pd.DataFrame(results)
            st.subheader("Results")
            st.dataframe(df_results, use_container_width=True)

            # Show accuracy preview
            valid_rows = df_results[df_results["true_label"].str.len() > 0]
            if len(valid_rows) > 0:
                accuracy = (valid_rows["true_label"] == valid_rows["pred_label"]).mean()
                st.info(f"📊 Preview Accuracy: {accuracy*100:.2f}% (based on {len(valid_rows)} files with detected true labels)")

            # CSV Download
            csv_bytes = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download CSV Results",
                data=csv_bytes,
                file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

            if show_heatmaps and len(gallery) > 0:
                st.subheader("Heatmap Gallery")
                for name, base_img, heat in gallery:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(base_img, caption=f"{name} - Original", use_container_width=True)
                    with c2:
                        if isinstance(heat, Image.Image):
                            st.image(heat, caption=f"{name} - Grad-CAM", use_container_width=True)
                        else:
                            st.info(f"{name}: {heat}")
# -------------- TAB 3: Dashboard --------------
with tabs[2]:
    st.header("Model Evaluation Dashboard")
    st.caption("Upload a CSV with columns: `image, true_label, pred_label` (and optionally per-class probabilities).")

    # Sample template download
    sample = pd.DataFrame({
        "image": ["img1.png", "img2.png", "img3.png", "img4.png"],
        "true_label": ["Glioma", "No Tumor", "Meningioma", "Pituitary"],
        "pred_label": ["Glioma", "No Tumor", "Glioma", "Pituitary"]
    })
    st.download_button(
        "📄 Download CSV Template",
        data=sample.to_csv(index=False).encode("utf-8"),
        file_name="evaluation_template.csv",
        mime="text/csv"
    )

    eval_csv = st.file_uploader("Upload evaluation CSV", type=["csv"])
    if eval_csv is not None:
        try:
            edf = pd.read_csv(eval_csv)
            
            # Debug: Show what columns are actually in the CSV
            st.write("📊 CSV Columns found:", list(edf.columns))
            st.write("📊 First few rows:")
            st.dataframe(edf.head())
            
            # Check for required columns with case-insensitive matching
            required_cols = {"true_label", "pred_label"}
            actual_cols = set(edf.columns.str.lower())
            
            if not required_cols.issubset(actual_cols):
                # Try to find case-insensitive matches
                col_mapping = {}
                for req_col in required_cols:
                    for actual_col in edf.columns:
                        if req_col.lower() in actual_col.lower():
                            col_mapping[req_col] = actual_col
                            break
                
                if len(col_mapping) == len(required_cols):
                    # Rename columns to standard names
                    edf = edf.rename(columns={v: k for k, v in col_mapping.items()})
                    st.success(f"✅ Auto-detected column mapping: {col_mapping}")
                else:
                    missing = required_cols - set(col_mapping.keys())
                    st.error(f"❌ Missing required columns: {missing}. CSV must include: true_label, pred_label")
                    st.stop()
            
            # Normalize labels and handle NaN values
            edf["true_label"] = edf["true_label"].astype(str).str.strip()
            edf["pred_label"] = edf["pred_label"].astype(str).str.strip()
            
            # Filter out rows with empty or invalid labels
            valid_mask = (edf["true_label"].str.len() > 0) & (edf["pred_label"].str.len() > 0)
            edf_valid = edf[valid_mask].copy()
            
            if len(edf_valid) == 0:
                st.error("❌ No valid rows found after filtering empty labels.")
                st.write("Rows with empty true_labels:", len(edf[edf["true_label"].str.len() == 0]))
                st.write("Rows with empty pred_labels:", len(edf[edf["pred_label"].str.len() == 0]))
                st.stop()
            
            st.subheader("Summary")
            accuracy = (edf_valid["true_label"] == edf_valid["pred_label"]).mean()
            total_samples = len(edf_valid)
            correct_predictions = (edf_valid["true_label"] == edf_valid["pred_label"]).sum()
            
            st.metric("Overall Accuracy", f"{accuracy*100:.2f}%")
            st.metric("Total Samples", total_samples)
            st.metric("Correct Predictions", correct_predictions)
            
            st.subheader("Per-Class Metrics")
            
            # Get unique labels from both true and pred columns
            all_labels = sorted(set(edf_valid["true_label"].unique()) | set(edf_valid["pred_label"].unique()))
            
            # Use actual class names found in data, but ensure our CLASS_NAMES are included
            labels = sorted(set(CLASS_NAMES + all_labels))
            
            # Confusion matrix
            cm = pd.crosstab(edf_valid["true_label"], edf_valid["pred_label"], 
                           rownames=["True"], colnames=["Pred"], dropna=False)
            
            # Reindex to include all possible labels
            cm = cm.reindex(index=labels, columns=labels, fill_value=0)
            
            st.write("Confusion Matrix")
            st.dataframe(cm.style.background_gradient(cmap="Blues"), use_container_width=True)

            # Per-class stats
            metrics = []
            for cls in labels:
                tp = cm.loc[cls, cls] if cls in cm.index and cls in cm.columns else 0
                fp = cm[cls].sum() - tp if cls in cm.columns else 0
                fn = cm.loc[cls].sum() - tp if cls in cm.index else 0
                
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                support = cm.loc[cls].sum() if cls in cm.index else 0
                
                metrics.append([cls, prec, rec, f1, support])

            mdf = pd.DataFrame(metrics, columns=["Class", "Precision", "Recall", "F1", "Support"])
            mdf[["Precision", "Recall", "F1"]] = mdf[["Precision", "Recall", "F1"]].applymap(lambda x: round(x * 100, 2))
            st.dataframe(mdf, use_container_width=True)

            st.subheader("Predicted Class Distribution")
            pred_dist = edf_valid["pred_label"].value_counts().reindex(labels, fill_value=0)
            st.bar_chart(pred_dist)
            
            st.subheader("True Class Distribution")
            true_dist = edf_valid["true_label"].value_counts().reindex(labels, fill_value=0)
            st.bar_chart(true_dist)

        except Exception as e:
            st.error(f"Failed to parse evaluation CSV: {e}")
            st.write("Error details:", str(e))
# -------------- TAB 4: About --------------
with tabs[3]:
    st.header("About This App")
    st.markdown("""
**Brain Tumor MRI Classifier**  
This app classifies MRI scans into four categories: **Glioma, Meningioma, No Tumor, Pituitary**,
uses **Grad-CAM** for explainability, supports **batch predictions**, and can generate a **PDF report**.

**Tech stack:** Streamlit, TensorFlow/Keras, NumPy, OpenCV, ReportLab.

> ⚠️ **Medical Disclaimer:** This tool is for educational & research purposes only and **must not** be used for clinical diagnosis.
""")
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("""
1. Go to **🔎 Predict (Single)** → upload an MRI → binary decision + subtype + optional Grad-CAM → optionally download a PDF report.  
2. Try **📦 Batch Predict** for multiple images & download a CSV of results (now with tumor flags).  
3. Open **📊 Dashboard** → upload an evaluation CSV (`image, true_label, pred_label`) to see metrics.  
    """)