"""
Streamlit Web App for Body-Part Classification with Grad-CAM Visualization

Predict which body part is shown in X-ray images and visualize model attention
using Grad-CAM from multiple ensemble backbones.

Usage
-----
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import io
import os


# ============================================================
#  Configuration
# ============================================================
IMG_SIZE = 224
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'weights', 'Ensemble_BodyParts_best.h5')
CLASS_LABELS = ["Elbow", "Hand", "Shoulder"]

BACKBONE_LAYERS = {
    'MobileNetV2': 'mob_conv_output',
    'DenseNet121': 'dense_conv_output',
    'InceptionV3': 'inc_conv_output',
}


# ============================================================
#  Preprocessing
# ============================================================
def preprocess_input(x):
    """Scale pixel values to [-1, 1]."""
    return x / 127.5 - 1.0


def load_and_preprocess(image_bytes):
    """
    Load image from bytes and return (original_uint8, preprocessed_batch).
    
    Parameters
    ----------
    image_bytes : bytes or PIL.Image
        Image data (uploaded file bytes or PIL Image)
        
    Returns
    -------
    original : np.ndarray uint8 (H, W, 3)
    batch : np.ndarray float32 (1, H, W, 3) in [-1, 1]
    """
    if isinstance(image_bytes, Image.Image):
        img = image_bytes.convert('RGB').resize((224, 224))
    else:
        img = load_img(io.BytesIO(image_bytes), target_size=(224, 224))
    
    img_array = img_to_array(img)                      # (300, 300, 3) float32 [0-255]
    original = img_array.astype(np.uint8)
    preprocessed = preprocess_input(img_array.copy())  # [-1, 1]
    batch = np.expand_dims(preprocessed, axis=0)       # (1, 300, 300, 3)
    return original, batch


# ============================================================
#  Grad-CAM Core
# ============================================================
def generate_gradcam(model, img_batch, conv_layer_name, pred_index=None):
    """
    Compute Grad-CAM heatmap.

    Parameters
    ----------
    model : tf.keras.Model
        The full ensemble model.
    img_batch : np.ndarray
        Preprocessed image batch of shape (1, 224, 224, 3).
    conv_layer_name : str
        Name of the conv-output layer to visualise.
    pred_index : int or None
        Class index to explain. None → use the top predicted class.

    Returns
    -------
    heatmap : np.ndarray (H, W) in [0, 1]
    """
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_batch)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)       # (1, h, w, c)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))   # (c,)

    conv_output = conv_output[0]                            # (h, w, c)
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]   # (h, w, 1)
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_img, heatmap, alpha=0.4, colormap='jet'):
    """
    Superimpose a Grad-CAM heatmap on the original image.

    Parameters
    ----------
    original_img : np.ndarray uint8 (H, W, 3)
    heatmap : np.ndarray (H, W) in [0, 1]
    alpha : float
        Transparency of the heatmap overlay (0-1)
    colormap : str
        matplotlib colormap name

    Returns
    -------
    superimposed : np.ndarray uint8 (H, W, 3)
    heatmap_colored : np.ndarray uint8 (H, W, 3)
    """
    # Resize heatmap to image size
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], (original_img.shape[0], original_img.shape[1])
    ).numpy().squeeze()

    # Colorize
    cmap = plt.colormaps[colormap]
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    superimposed = (heatmap_colored * alpha +
                    original_img * (1 - alpha)).astype(np.uint8)
    return superimposed, heatmap_colored


# ============================================================
#  Main Streamlit App
# ============================================================
st.set_page_config(page_title="Body-Part X-ray Classifier", layout="wide")

st.title("🏥 Body-Part X-ray Classifier")
st.markdown("---")
st.markdown("""
**Upload an X-ray image** to predict which body part it shows (Elbow, Hand, or Shoulder),
and visualize the model's prediction using Grad-CAM from multiple deep learning backbones.
""")
st.markdown("---")

# ────────────────────────────────────────────────────────
#  Load Model (with caching)
# ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the ensemble model (cached)."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found at {MODEL_PATH}")
        st.error(f"Please ensure the model file exists in the `weights/` directory.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()
st.success("✅ Model loaded successfully!")
st.markdown("---")

# ────────────────────────────────────────────────────────
#  Image Upload
# ────────────────────────────────────────────────────────
st.sidebar.header("📤 Upload Image")
uploaded_file = st.sidebar.file_uploader(
    "Choose an X-ray image (JPG, PNG):",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, PNG"
)

if uploaded_file is not None:
    # ────────────────────────────────────────────────────────
    #  Process Image & Make Prediction
    # ────────────────────────────────────────────────────────
    original, batch = load_and_preprocess(uploaded_file.read())
    
    # Get prediction
    predictions = model.predict(batch, verbose=0)
    pred_class = int(np.argmax(predictions[0]))
    pred_label = CLASS_LABELS[pred_class]
    pred_conf = float(predictions[0][pred_class]) * 100
    
    # Get confidences for all classes
    all_confs = {CLASS_LABELS[i]: float(predictions[0][i]) * 100 for i in range(len(CLASS_LABELS))}
    
    # ────────────────────────────────────────────────────────
    #  Display Original Image & Prediction
    # ────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Original Image")
        st.image(original, use_container_width=True, clamp=True)
    
    with col2:
        st.subheader("🎯 Prediction")
        
        # Show predicted class prominently
        st.markdown(f"### **Predicted Body Part: {pred_label}**")
        st.progress(pred_conf / 100.0)
        st.markdown(f"**Confidence: {pred_conf:.2f}%**")
        
        st.markdown("---")
        st.markdown("### Confidence by Class:")
        for label, conf in all_confs.items():
            st.write(f"- **{label}**: {conf:.2f}%")
    
    st.markdown("---")
    
    # ────────────────────────────────────────────────────────
    #  Grad-CAM Visualization
    # ────────────────────────────────────────────────────────
    st.subheader("🔍 Grad-CAM Attention Maps")
    st.markdown("""
    Grad-CAM shows which regions of the image the model focused on to make its prediction.
    Each visualization comes from a different deep learning backbone:
    """)
    
    # Options for visualization
    col1_viz, col2_viz, col3_viz = st.columns(3)
    show_all = st.checkbox("Show all backbones side-by-side", value=True)
    
    if show_all:
        # Create figure with all backbones
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f"Grad-CAM Attention Maps — Predicted: {pred_label} ({pred_conf:.1f}%)",
                     fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(original)
        axes[0, 0].set_title("Original Image", fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Generate Grad-CAM for each backbone
        for idx, (backbone_name, layer_name) in enumerate(BACKBONE_LAYERS.items()):
            row = (idx + 1) // 2
            col = (idx + 1) % 2
            
            heatmap = generate_gradcam(model, batch, layer_name, pred_class)
            overlay, _ = overlay_heatmap(original, heatmap, alpha=0.4)
            
            axes[row, col].imshow(overlay)
            axes[row, col].set_title(f"Grad-CAM: {backbone_name}", fontsize=12, fontweight='bold')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
    else:
        # Single backbone selection
        selected_backbone = st.selectbox(
            "Select a backbone to visualize:",
            list(BACKBONE_LAYERS.keys())
        )
        
        layer_name = BACKBONE_LAYERS[selected_backbone]
        heatmap = generate_gradcam(model, batch, layer_name, pred_class)
        overlay, heatmap_colored = overlay_heatmap(original, heatmap, alpha=0.4)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Grad-CAM — {selected_backbone} | Body Part: {pred_label} ({pred_conf:.1f}%)",
                     fontsize=14, fontweight='bold')
        
        axes[0].imshow(original)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap_colored)
        axes[1].set_title('Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Heatmap Overlay', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    st.markdown("---")
    
    # ────────────────────────────────────────────────────────
    #  Download Prediction Results
    # ────────────────────────────────────────────────────────
    st.subheader("📥 Export Results")
    
    # Create a text report
    report = f"""
═══════════════════════════════════════════
     BODY-PART CLASSIFICATION REPORT
═══════════════════════════════════════════

PREDICTION RESULT:
  Predicted Body Part: {pred_label}
  Confidence: {pred_conf:.2f}%

CLASS PROBABILITIES:
  • Elbow: {all_confs['Elbow']:.2f}%
  • Hand: {all_confs['Hand']:.2f}%
  • Shoulder: {all_confs['Shoulder']:.2f}%

MODEL INFORMATION:
  • Architecture: Ensemble (MobileNetV2 + DenseNet121 + InceptionV3)
  • Input Size: 224×224 pixels
  • Classes: 3 (Elbow, Hand, Shoulder)
  • Visualization: Grad-CAM (multiple backbones)

═══════════════════════════════════════════
"""
    
    st.download_button(
        label="📄 Download Prediction Report (TXT)",
        data=report,
        file_name="prediction_report.txt",
        mime="text/plain"
    )

else:
    st.info("👆 **Upload an X-ray image from the sidebar to get started!**")
    st.markdown("""
    ### 📋 How to use this app:
    1. Click on the upload button in the sidebar
    2. Select an X-ray image (JPG or PNG format)
    3. The model will automatically predict which body part is shown
    4. View the Grad-CAM attention maps to see which regions influenced the prediction
    5. Export your results as a text report
    
    ### 🎯 Supported Body Parts:
    - **Elbow** - Arm joint injuries
    - **Hand** - Hand and wrist fractures
    - **Shoulder** - Shoulder and upper arm injuries
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### ℹ️ About
**Body-Part X-ray Classifier**

An AI-powered system for classifying X-ray images by body part,
powered by an ensemble of deep learning models with Grad-CAM
explainability.

- **Model**: Ensemble (MobileNetV2, DenseNet121, InceptionV3)
- **Task**: 3-way body-part classification
- **Explainability**: Grad-CAM attention maps
""")


