"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for the ensemble model.

Supports visualising attention from any of the three backbones:
    • MobileNetV2   (layer name: mob_conv_output)
    • DenseNet121   (layer name: dense_conv_output)
    • InceptionV3   (layer name: inc_conv_output)

Usage
-----
    # From terminal
    python grad_cam.py --image test_xray.png \
                       --model weights/Ensemble_Elbow_frac.h5 \
                       --backbone all

    # From code
    from grad_cam import generate_gradcam, overlay_heatmap
    heatmap = generate_gradcam(model, img_array, 'mob_conv_output')
    overlay = overlay_heatmap(original_img, heatmap)
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# ============================================================
#  Constants
# ============================================================
BACKBONE_LAYERS = {
    'efficientnetv2s': 'eff_conv_output',
    'densenet121': 'dense_conv_output',
    'inceptionv3': 'inc_conv_output',
    # Legacy alias for old models
    'mobilenetv2': 'mob_conv_output',
}

IMG_SIZE = (300, 300)


# ============================================================
#  Preprocessing  (must match training)
# ============================================================
def preprocess_input(x):
    return x / 127.5 - 1.0


def load_and_preprocess(image_path):
    """Load an image and return (original_uint8, preprocessed_batch)."""
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)                      # (224, 224, 3) float32 [0-255]
    original = img_array.astype(np.uint8)
    preprocessed = preprocess_input(img_array.copy())   # [-1, 1]
    batch = np.expand_dims(preprocessed, axis=0)        # (1, 224, 224, 3)
    return original, batch


# ============================================================
#  Grad-CAM core
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
        Name of the conv-output layer to visualise
        (mob_conv_output | dense_conv_output | inc_conv_output).
    pred_index : int or None
        Class index to explain.  None → use the top predicted class.

    Returns
    -------
    heatmap : np.ndarray   (H, W) in [0, 1]
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


# ============================================================
#  Overlay helper
# ============================================================
def overlay_heatmap(original_img, heatmap, alpha=0.4, colormap='jet'):
    """
    Superimpose a Grad-CAM heatmap on the original image.

    Returns
    -------
    superimposed : np.ndarray  uint8 (H, W, 3)
    heatmap_colored : np.ndarray  uint8 (H, W, 3)
    """
    # Resize heatmap to image size
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], (original_img.shape[0], original_img.shape[1])
    ).numpy().squeeze()

    # Colourize
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]       # drop alpha
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    superimposed = (heatmap_colored * alpha +
                    original_img * (1 - alpha)).astype(np.uint8)
    return superimposed, heatmap_colored


# ============================================================
#  Visualisation
# ============================================================
def visualise_all_backbones(model, image_path, save_dir=None):
    """
    Generate Grad-CAM from every backbone and show side-by-side.
    """
    original, batch = load_and_preprocess(image_path)
    predictions = model.predict(batch, verbose=0)
    pred_class = int(np.argmax(predictions[0]))
    pred_conf  = float(predictions[0][pred_class]) * 100

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Original image
    axes[0].imshow(original)
    axes[0].set_title(f'Original\nPred: class {pred_class} ({pred_conf:.1f}%)',
                      fontsize=12)
    axes[0].axis('off')

    backbone_names = list(BACKBONE_LAYERS.keys())
    for idx, name in enumerate(backbone_names):
        layer_name = BACKBONE_LAYERS[name]
        heatmap = generate_gradcam(model, batch, layer_name, pred_class)
        overlay, _ = overlay_heatmap(original, heatmap)
        axes[idx + 1].imshow(overlay)
        axes[idx + 1].set_title(f'Grad-CAM: {name}', fontsize=12)
        axes[idx + 1].axis('off')

    plt.suptitle('Ensemble Grad-CAM Comparison', fontsize=15, y=1.02)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.splitext(os.path.basename(image_path))[0]
        out = os.path.join(save_dir, f'gradcam_{fname}.jpeg')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  ✔ Saved → {out}")
    else:
        plt.show()
    plt.close(fig)


def visualise_single_backbone(model, image_path, backbone='mobilenetv2',
                               save_dir=None):
    """Generate Grad-CAM for one specific backbone."""
    original, batch = load_and_preprocess(image_path)
    predictions = model.predict(batch, verbose=0)
    pred_class = int(np.argmax(predictions[0]))
    pred_conf  = float(predictions[0][pred_class]) * 100

    layer_name = BACKBONE_LAYERS[backbone]
    heatmap = generate_gradcam(model, batch, layer_name, pred_class)
    overlay, heatmap_colored = overlay_heatmap(original, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original);        axes[0].set_title('Original', fontsize=12)
    axes[1].imshow(heatmap_colored); axes[1].set_title('Heatmap', fontsize=12)
    axes[2].imshow(overlay);         axes[2].set_title(
        f'Overlay — {backbone}\nclass {pred_class} ({pred_conf:.1f}%)', fontsize=12)
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.splitext(os.path.basename(image_path))[0]
        out = os.path.join(save_dir, f'gradcam_{backbone}_{fname}.jpeg')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  ✔ Saved → {out}")
    else:
        plt.show()
    plt.close(fig)


# ============================================================
#  CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Grad-CAM visualisation for the ensemble bone-fracture model')
    parser.add_argument('--image', required=True, help='Path to X-ray image')
    parser.add_argument('--model', required=True, help='Path to saved .h5 model')
    parser.add_argument('--backbone', default='all',
                        choices=['efficientnetv2s', 'densenet121', 'inceptionv3', 'mobilenetv2', 'all'],
                        help='Which backbone to visualise (default: all)')
    parser.add_argument('--save_dir', default='gradcam_results',
                        help='Directory to save output images')
    args = parser.parse_args()

    print(f"\n  Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model)

    if args.backbone == 'all':
        visualise_all_backbones(model, args.image, args.save_dir)
    else:
        visualise_single_backbone(model, args.image, args.backbone, args.save_dir)

    print("  ✅ Grad-CAM complete!\n")


if __name__ == '__main__':
    main()
