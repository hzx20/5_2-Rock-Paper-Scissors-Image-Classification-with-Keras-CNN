# Rock-Paper-Scissors Image Classification with Keras CNN

An experiment project for Rock-Paper-Scissors hand gesture image classification using Convolutional Neural Networks (CNN) built with Keras 3 and PyTorch backend.

## Project Overview

This project implements a complete deep learning pipeline:

1. **Data Preparation** - Download and verify the RPS image dataset
2. **Model Training** - Build and train a CNN model using Keras
3. **Model Evaluation** - Generate performance metrics and visualizations

## Dataset

The experiment uses the [Laurence Moroney's Rock-Paper-Scissors dataset](https://storage.googleapis.com/learning-datasets/):

- **Training Set**: 2,520 images (840 per class)
- **Test Set**: 372 images (124 per class)
- **Image Dimensions**: 300x300 pixels, RGB
- **Classes**: `rock`, `paper`, `scissors`

## Model Architecture

```
Input (3, 48, 48)
  ├── Conv2D (32 filters, 3x3) -> ReLU -> MaxPool2D
  ├── Conv2D (64 filters, 3x3) -> ReLU -> MaxPool2D
  ├── Conv2D (64 filters, 3x3) -> ReLU -> MaxPool2D
  ├── Flatten
  ├── Dropout (0.5)
  ├── Dense (64) -> ReLU
  └── Dense (3) -> Softmax
```

Total parameters: ~204K

## Results

| Metric | Train | Validation | Test |
|--------|-------|-----------|------|
| Accuracy | ~94% | ~46% | ~46% |
| Loss | 0.34 | 1.50 | 1.32 |

Note: The model shows signs of overfitting. Performance can be improved with more epochs, data augmentation, or larger image sizes.

## Requirements

- Python 3.10+
- keras >= 3.0.0
- torch >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- Pillow >= 10.0.0
- scikit-learn >= 1.3.0

## Installation

```bash
# Clone the repository
git clone https://github.com/hzx20/Rock-Paper-Scissors-Image-Classification-with-Keras-CNN.git
cd Rock-Paper-Scissors-Image-Classification-with-Keras-CNN

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Step 1: Download dataset

```bash
python download_data.py
```

This will download the dataset ZIP files and extract them to the `rps_data/` directory.

### Step 2: Run the complete experiment

```bash
python rps_experiment.py
```

This will:

1. Verify the dataset and generate sample image previews
2. Build and train the CNN model
3. Generate performance visualizations (training curves, confusion matrix, prediction examples)
4. Save the trained model to `results/rps_model.keras`

All results will be saved to the `results/` directory:

- `results/01_sample_images.png` - Sample images from the dataset
- `results/02_training_curves.png` - Training/validation accuracy and loss curves
- `results/03_predictions.png` - Example predictions on test images
- `results/04_confusion_matrix.png` - Confusion matrix visualization
- `results/classification_report.txt` - Detailed classification metrics
- `results/rps_model.keras` - Trained model file

## Project Structure

```
.
├── download_data.py          # Dataset download script
├── rps_experiment.py         # Main experiment script (all 3 steps)
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── rps_data/                # Dataset (downloaded on first run)
│   ├── rps/                 # Training set
│   │   ├── paper/
│   │   ├── rock/
│   │   └── scissors/
│   └── rps-test-set/        # Test set
│       ├── paper/
│       ├── rock/
│       └── scissors/
└── results/                 # Experiment output
    ├── 01_sample_images.png
    ├── 02_training_curves.png
    ├── 03_predictions.png
    ├── 04_confusion_matrix.png
    ├── classification_report.txt
    └── rps_model.keras
```

## How to Improve

- **Increase image size** (e.g., 150x150 instead of 48x48)
- **Add data augmentation** (rotation, horizontal flip, zoom, translation)
- **Train for more epochs** (15-30 epochs instead of 5)
- **Use a deeper model** (more convolutional layers)
- **Apply transfer learning** (use pre-trained models like VGG16, MobileNet)
- **Tune hyperparameters** (learning rate, batch size, dropout rate)

## References

- [TensorFlow/Keras Documentation](https://keras.io/)
- [Rock Paper Scissors Dataset](https://storage.googleapis.com/learning-datasets/)

## License

This project is for educational purposes.
