# 🏥 Body-Part X-ray Classification with Grad-CAM

An interactive Streamlit web application for predicting body parts in X-ray images using an ensemble deep learning model with Grad-CAM explainability.

## 📋 Features

- **Image Upload**: Upload X-ray images (JPG, PNG)
- **Body-Part Classification**: Predicts Elbow, Hand, or Shoulder
- **Confidence Scores**: Shows prediction confidence for all classes
- **Grad-CAM Visualization**: Explainability maps from multiple backbones:
  - EfficientNetV2S
  - DenseNet121
  - InceptionV3
- **Export Results**: Download prediction reports

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Model Files

Ensure you have the trained model weights in the `weights/` directory:
```
weights/
  ├── Ensemble_BodyParts_best.h5
  └── Ensemble_BodyParts.h5
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

## 📁 Project Structure

```
boneFracture Dataset/
├── app.py                              # Main Streamlit application
├── ensemble_training_parts.py          # Model training script
├── grad_cam.py                         # Grad-CAM visualization utilities
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── Dataset/                            # Training/test data
│   ├── train_valid/
│   │   ├── Elbow/
│   │   ├── Hand/
│   │   └── Shoulder/
│   └── test/
│       ├── Elbow/
│       ├── Hand/
│       └── Shoulder/
├── weights/                            # Trained model weights
│   ├── Ensemble_BodyParts_best.h5
│   └── Ensemble_BodyParts.h5
└── plots/                              # Training visualizations
    ├── Ensemble_BodyPart_Accuracy.jpeg
    ├── Ensemble_BodyPart_Loss.jpeg
    └── Ensemble_BodyPart_CM.jpeg
```

## 🔧 How the Model Works

### Architecture

The model is an **3-backbone ensemble**:
1. **EfficientNetV2S** - Efficient feature extraction
2. **DenseNet121** - Dense connections for gradient flow
3. **InceptionV3** - Multi-scale feature learning

All three backbones process the input image independently, extract features, and their outputs are concatenated and passed through a dense head for final classification.

### Training Process

The model is trained in **two phases**:

1. **Phase 1 - Head Warm-up** (30 epochs)
   - Freeze backbone weights
   - Train the new classification head
   - Build initial performance

2. **Phase 2 - Fine-tuning** (80 epochs)
   - Unfreeze top 30% of backbone layers
   - Train entire model with low learning rate
   - Use cosine annealing for learning rate scheduling
   - Apply early stopping based on validation accuracy

### Input Preprocessing

- **Image Size**: 224×224 pixels
- **Scaling**: Normalize to [-1, 1]
- **Augmentation**: 
  - Horizontal flips
  - Rotations (±20°)
  - Zoom (0.8-1.2×)
  - Brightness changes (0.7-1.3×)
  - Cutout regularization

## 🔍 Grad-CAM Explanation

Grad-CAM (Gradient-weighted Class Activation Mapping) shows which regions of the image the model focused on when making its prediction.

### What the Heatmaps Show

- **Red/Yellow regions**: High importance for the prediction
- **Blue regions**: Low importance for the prediction
- **Helps verify** if the model is focusing on the correct anatomical regions

### How It Works

1. Forward pass: Image → Model → Prediction
2. Compute gradients of the predicted class w.r.t. feature maps
3. Weight feature map activations by gradients
4. Aggregate to create attention heatmap
5. Overlay on original image for visualization

## 📊 Model Performance

### Trained Classes

- **Elbow**: Elbow and forearm fractures
- **Hand**: Hand, wrist, and finger fractures
- **Shoulder**: Shoulder and upper arm fractures

### Metrics

See `plots/Ensemble_BodyPart_CM.jpeg` for confusion matrix and accuracy/loss curves.

## 💡 Usage Tips

1. **Best Results**: Use clear, well-captured X-ray images
2. **Image Quality**: Images should clearly show the body part
3. **Format**: JPG or PNG, any resolution (will be resized to 300×300)
4. **Confidence Score**: Higher confidence indicates more certain predictions
5. **Grad-CAM**: Use attention maps to understand model decisions

## 🔄 Retraining the Model

To retrain from scratch with your dataset:

```bash
python ensemble_training_parts.py
```

This will:
- Load all images from `Dataset/train_valid/` and `Dataset/test/`
- Train the ensemble model in two phases
- Save best weights to `weights/Ensemble_BodyParts_best.h5`
- Generate training curves and confusion matrix in `plots/`

## 📝 Generating Grad-CAM Offline

To generate Grad-CAM visualizations from command line:

```bash
# All backbones side-by-side
python grad_cam.py --image path/to/image.jpg \
                   --model weights/Ensemble_BodyParts_best.h5 \
                   --backbone all

# Single backbone
python grad_cam.py --image path/to/image.jpg \
                   --model weights/Ensemble_BodyParts_best.h5 \
                   --backbone densenet121
```

## 🛠️ Troubleshooting

### Model not found error
- Ensure `weights/Ensemble_BodyParts_best.h5` exists
- Check file path is correct
- Run training script if models are missing

### Out of memory error
- Reduce image batch size in `app.py`
- Close other applications
- Use GPU acceleration if available (TensorFlow will auto-detect)

### Streamlit slow to load
- Model is cached after first load
- First prediction may take 10-30 seconds
- Subsequent predictions should be instant

## 📚 Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Web app framework |
| tensorflow | Deep learning framework |
| numpy | Numerical computing |
| matplotlib | Visualization |
| pillow | Image processing |
| scikit-learn | Metrics and utilities |

## 📄 License

This project uses medical X-ray data for bone fracture classification.

## 👨‍💻 Author Notes

- The ensemble approach combines strengths of three different architectures
- Grad-CAM provides interpretability for medical imaging decisions
- The model is trained with class balancing and regularization for robustness
- Two-phase training ensures both head initialization and backbone fine-tuning

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the training logs in `plots/`
3. Verify input image format and quality
4. Ensure all dependencies are installed correctly

---

**Last updated**: March 2026
