import streamlit as st
import numpy as np
import cv2

from components.predictor import predict_ui_elements
from components.ocr_add import add_ocr_data
from components.embed_add import enrich_components
from components.matcher import hungarian_match_components
from components.mismatch_analysis import classify_mismatches, strip_all_embeddings
from components.mismatch_heuristics import enrich_with_heuristics
from components.theme_check import check_theme_difference
from components.visualise import visualize_report_differences
from components.semantic import semantic_summary
from components.explainer import explain_ui_mismatches

st.set_page_config(page_title="⚙️ Diffy UI Validator", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: #fff0f6;
    color: #333;
}
h1 {
    color: #e75480;
}
.stButton>button {
    background-color: #e75480;
    color: white;
    border-radius: 8px;
    padding: 8px 24px;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: #c1416d;
}
</style>
""", unsafe_allow_html=True)

st.title("⚙️ Diffy! Validate your screenshot in a click")
st.markdown("Upload a **Golden** and an **Actual** screenshot to visually highlight UI component differences.")

col_golden, col_actual = st.columns(2)
with col_golden:
    golden_file = st.file_uploader("Golden Screenshot", type=["png", "jpg", "jpeg"])
with col_actual:
    actual_file = st.file_uploader("Actual Screenshot", type=["png", "jpg", "jpeg"])

def load_image(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if golden_file and actual_file:
    golden_img = load_image(golden_file)
    actual_img = load_image(actual_file)

    with st.spinner("🔍 Processing screenshots..."):
        progress_bar = st.progress(0)

        components_golden = predict_ui_elements(golden_img)
        progress_bar.progress(20)

        components_actual = predict_ui_elements(actual_img)
        progress_bar.progress(40)

        components_golden = add_ocr_data(golden_img, components_golden)
        components_actual = add_ocr_data(actual_img, components_actual)
        progress_bar.progress(60)

        image_shape = golden_img.shape
        components_golden = enrich_components(golden_img, components_golden, image_shape)
        components_actual = enrich_components(actual_img, components_actual, image_shape)
        progress_bar.progress(80)

        results = hungarian_match_components(components_golden, components_actual)
        results = classify_mismatches(results)
        enrich_with_heuristics(results["matches"], golden_img, actual_img)

        report = {
            **results,
            "missing_components": results.get("unmatched_golden", []),
            "extra_components": results.get("unmatched_actual", [])
        }

        check_theme_difference(golden_img, actual_img, report)
        strip_all_embeddings(results)

        progress_bar.progress(100)

    st.success("✅ Done! Visual mismatch overlays are ready.")
    st.subheader("🖼️ Annotated Comparison")

    golden_overlay, actual_overlay = visualize_report_differences(golden_img, actual_img, report)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Golden Screenshot (Annotated)**")
        st.image(golden_overlay, use_container_width=True)

    with col2:
        st.markdown("**Actual Screenshot (Annotated)**")
        st.image(actual_overlay, use_container_width=True)

    try:
        with st.spinner("📖 Generating LLM explanation..."):
            semantic_summary = semantic_summary(actual_img)
            explanation = explain_ui_mismatches(
                semantic_meaning=semantic_summary,
                results=report,
                golden_img=golden_img,
                actual_img=actual_img
            )
            st.subheader("🧾 Mismatch Explanation")
            st.markdown(explanation)
    except Exception as e:
        st.error(f"❌ Failed to generate explanation: {str(e)}")

else:
    st.info("📂 Please upload both screenshots above to get started.")
