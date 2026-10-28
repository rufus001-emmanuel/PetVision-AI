import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
# IMPORTANT: change these to match your existing notebook.
MODEL_PATH = "model.keras"
LABELS = ["dog", "cat", "neither"]  # Use your exact class-index order.
IMAGE_SIZE = (160, 160)             # Use your model's actual input size.
PIXEL_SCALE = "0_1"                 # "0_1" or "0_255"; match training.
st.set_page_config(
    page_title="PetVision AI",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Front-end: navy, white, cool gray, restrained blue and teal.
st.markdown("""
<style>
.stApp {background:#F4F7FB;color:#172334}
[data-testid="stHeader"] {background:rgba(244,247,251,.92)}
[data-testid="stSidebar"] {background:#14243A}
[data-testid="stSidebar"] * {color:#F5F8FC !important}
.block-container {padding-top:1.5rem;padding-bottom:2.5rem;max-width:1400px}
.pv-hero {background:#14243A;color:white;padding:1.35rem 1.5rem;
border-radius:18px;margin-bottom:1rem;border:1px solid #263B56}
.pv-eyebrow {color:#7DD3C7;font-size:.78rem;letter-spacing:.12em;
text-transform:uppercase;font-weight:700}
.pv-hero h1 {color:white;margin:.25rem 0 .35rem;font-size:2rem}
.pv-hero p {color:#D8E2EF;margin:0}
.pv-card {background:white;border:1px solid #DFE7F0;border-radius:15px;
padding:1rem 1.1rem;box-shadow:0 3px 12px rgba(20,36,58,.04)}
div[data-testid="stMetric"] {background:white;border:1px solid #DFE7F0;
padding:.9rem 1rem;border-radius:14px}
.stButton button,.stDownloadButton button {border-radius:10px}
h2,h3 {color:#14243A}
@media(max-width:700px) {
.block-container {padding-left:1rem;padding-right:1rem}
.pv-hero h1 {font-size:1.55rem}
}
</style>
""", unsafe_allow_html=True)
@st.cache_resource
def load_model():
     return tf.keras.models.load_model(MODEL_PATH)
def prepare_image(pil_image):
      image = pil_image.convert("RGB").resize(IMAGE_SIZE)
      arr = np.asarray(image, dtype=np.float32)
      if PIXEL_SCALE == "0_1":
           arr = arr / 255.0
       # If your notebook uses special preprocessing, reproduce it here.
      return np.expand_dims(arr, axis=0)
def predict_image(pil_image, model):
    output = model.predict(prepare_image(pil_image), verbose=0)
    scores = np.asarray(output)[0].astype(float).ravel()
    if len(scores) != len(LABELS):
        raise ValueError(
            f"Model returned {len(scores)} outputs but LABELS has "
            f"{len(LABELS)} labels. Check the class count and order."
        )
    index = int(np.argmax(scores))
    return LABELS[index], float(scores[index]), scores
def read_csv_if_exists(path):
    return pd.read_csv(path) if Path(path).exists() else None
with st.sidebar:
    st.markdown("## ■ PetVision AI")
    st.caption("Image classification dashboard")
    page = st.radio(
         "NAVIGATION",
          ["Overview", "Predict image", "Batch analysis", "Model evaluation"],
    )
    st.divider()
    st.caption("Verify the model settings against the notebook.")
st.markdown("""
<div class="pv-hero">
 <div class="pv-eyebrow">Computer vision · Image classification</div>
 <h1>PetVision AI</h1>
 <p>Upload an image, review a prediction, and analyse multiple images in one place.</p>
</div>
""", unsafe_allow_html=True)
if page == "Overview":
     col1, col2, col3 = st.columns(3)
     col1.metric("Workflow", "Image → prediction")
     col2.metric("Batch tools", "CSV export")
     col3.metric("Evaluation", "Test metrics")
     st.markdown("### Welcome")
     st.markdown(
         '<div class="pv-card"><b>How to use this dashboard</b><br>'
          'Choose Predict image for one file, Batch analysis for multiple files, '
          'or Model evaluation for genuine test-set results.</div>',
           unsafe_allow_html=True,
     )
     st.info("Verify class order, image size, and preprocessing before demonstration.")
elif page == "Predict image":
    st.markdown("### Predict a single image")
    uploaded = st.file_uploader(
        "Choose JPG, JPEG, or PNG", type=["jpg", "jpeg", "png"]
    )
    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
            left, right = st.columns([1, 1.2], gap="large")
            with left:
                st.image(image, caption="Uploaded image", use_container_width=True)
            with right:
                model = load_model()
                label, score, scores = predict_image(image, model)
                st.markdown('<div class="pv-card">', unsafe_allow_html=True)
                st.markdown("#### Prediction")
                st.markdown(f"## {label.title()}")
                st.metric("Top model score", f"{score * 100:.2f}%")
                st.caption("This score is not a guarantee of correctness.")
                score_table = pd.DataFrame({
                    "Class": LABELS,
                    "Score (%)": np.round(scores * 100, 2),
                })
                st.dataframe(score_table, hide_index=True, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        except Exception as error:
                st.error(f"Could not process this image: {error}")
                st.warning("Check model file, class order, image size, and preprocessing.")
    else:
      st.info("Choose an image to see its preview and prediction.")
elif page == "Batch analysis":
        st.markdown("### Batch image analysis")
        files = st.file_uploader(
           "Select multiple images (20+ supported)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        if files:
            st.write(f"**{len(files)} image(s) selected**")
            model = load_model()
            rows = []
            columns = st.columns(4)
            for i, file in enumerate(files):
                try:
                   image = Image.open(file).convert("RGB")
                   label, score, scores = predict_image(image, model)
                   row = {
                      "filename": file.name,
                      "prediction": label,
                      "top_score": round(score, 6),
                  }
                  for name, value in zip(LABELS, scores):
                      row[f"score_{name}"] = round(float(value), 6)
                  rows.append(row)
                  with columns[i % 4]:
                      st.image(image, use_container_width=True)
                      st.caption(file.name)
                      st.write(f"**{label.title()}**")
                      st.caption(f"Score: {score * 100:.1f}%")
     except Exception as error:
        rows.append({
          "filename": file.name,
          "prediction": "ERROR",
          "error": str(error),
      })
  results = pd.DataFrame(rows)
  st.markdown("### Results")
  st.dataframe(results, use_container_width=True, hide_index=True)
  st.download_button(
      "Download predictions as CSV",
       data=results.to_csv(index=False).encode("utf-8"),
       file_name="petvision_batch_predictions.csv",
       mime="text/csv",
  )
  st.caption("Batch predictions are not labelled evaluation results.")
else:
  st.info("Select multiple images to create a batch report.")
elif page == "Model evaluation":
    st.markdown("### Model evaluation")
    st.write("These files must come from labelled test data and actual predictions.")
    metrics_file = Path("evaluation/evaluation_metrics.json")
    if metrics_file.exists():
         try:
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            columns = st.columns(4)
            metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
            for column, key in zip(columns, metric_keys):
                value = metrics.get(key)
                display = f"{float(value):.3f}" if value is not None else "N/A"
                column.metric(key.replace("_", " ").title(), display)
        except Exception as error:
             st.error(f"Cannot read evaluation_metrics.json: {error}")
    else:
        st.warning("Missing evaluation/evaluation_metrics.json.")
    report = read_csv_if_exists("evaluation/classification_report.csv")
    if report is not None:
         st.markdown("#### Per-class report")
         st.dataframe(report, use_container_width=True, hide_index=True)
    else:
        st.info("Add evaluation/classification_report.csv for per-class metrics.")
    matrix = read_csv_if_exists("evaluation/confusion_matrix.csv")
    if matrix is not None:
         st.markdown("#### Confusion matrix")
         st.dataframe(matrix, use_container_width=True, hide_index=True)
         numeric = matrix.select_dtypes(include="number")
         if not numeric.empty:
              st.bar_chart(numeric)
else:
    st.info("Add evaluation/confusion_matrix.csv for the confusion matrix.")
st.caption("Never fill these evaluation files with guessed performance values.")
