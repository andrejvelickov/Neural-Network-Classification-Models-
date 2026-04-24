# Neural Network Classification Models

A collection of neural network classification projects built with **TensorFlow/Keras**, developed as a university project at ETF Belgrade. The repository contains two models: an image classification CNN for tennis player action recognition, and a tabular classification FCNN for rice variety identification.

***

## Repository Structure

```
Neural-Network-Classification-Models/
├── cnn/
│   ├── main.py          # CNN model — training, evaluation, visualization
│   └── split_data.py    # Dataset splitting utility (train/val/test)
├── fcnn/
│   ├── main.py          # FCNN model — hyperparameter tuning, evaluation
│   └── data/
│       └── Rice.csv     # Rice variety tabular dataset
└── .gitignore
```

> **Note:** The CNN dataset (`Tennis Player Actions Dataset for Human Pose Estimation`) is not included in the repo due to size. Download it separately (e.g. from Kaggle) and place it under `cnn/data/`.

***

## Models

### CNN — Tennis Player Action Classification

An image classification model that recognizes **tennis player pose/action classes** from photos.

| Property | Value |
|----------|-------|
| Task | Multi-class image classification |
| Dataset | Tennis Player Actions Dataset for Human Pose Estimation |
| Input size | 64×64 RGB images |
| Architecture | 4× Conv2D → MaxPooling → Dropout(0.2) → Flatten → Dense(128) → Softmax |
| Optimizer | Adam (lr=0.001) |
| Loss | Sparse Categorical Crossentropy |
| Regularization | Data augmentation (flip, rotation, zoom), early stopping (patience=40) |
| Epochs | Up to 250 |

**Architecture overview:**
```
Input (64×64×3)
  → Data Augmentation (RandomFlip, RandomRotation, RandomZoom)
  → Rescaling (1/255)
  → Conv2D(16) → MaxPool
  → Conv2D(32) → MaxPool
  → Conv2D(64) → MaxPool
  → Conv2D(128) → Dropout(0.2) → Flatten
  → Dense(128, relu)
  → Dense(num_classes, softmax)
```

**Outputs:** training/validation accuracy & loss curves, confusion matrices for train and test sets, and sample correctly/incorrectly classified images.

***

### FCNN — Rice Variety Classification

A fully-connected neural network that classifies rice grains into one of three varieties based on morphological features from a CSV dataset.

| Property | Value |
|----------|-------|
| Task | Multi-class tabular classification |
| Dataset | `Rice.csv` — morphological grain features |
| Classes | Cammeo, Kecimen, Osmancik |
| Architecture | Dense(units) → Dense(12) → Dense(3, softmax) |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Tuning | GridSearchCV over `units` ∈ {16, 32} and `epochs` ∈ {50, 100} |
| Regularization | Class weights (balanced), early stopping (patience=5) |

**Pipeline:**
```
Rice.csv
  → StandardScaler (feature normalization)
  → LabelEncoder + to_categorical (one-hot labels)
  → Train/Val/Test split (60% / 20% / 20%)
  → GridSearchCV (3-fold CV, keras_tuner)
  → Best model refit with class weights
  → Confusion matrix + per-class metrics
```

**Outputs:** loss curve, confusion matrices for train and test sets, per-class accuracy, precision, recall, and F1 score.

***

## Prerequisites

```bash
pip install tensorflow keras keras-tuner scikit-learn scikeras pandas numpy matplotlib
```

Or install from a requirements file if provided:
```bash
pip install -r requirements.txt
```

***

## Running

### CNN
```bash
cd cnn
# Place dataset at: ./data/Tennis Player Actions Dataset for Human Pose Estimation/
python main.py
```

### FCNN
```bash
cd fcnn
# Dataset already included at: ./data/Rice.csv
python main.py
```

***

## Evaluation Metrics

Both models report the following per-class metrics on the test set:

- **Accuracy** — (TP + TN) / total
- **Precision** — TP / (TP + FP)
- **Recall (Sensitivity)** — TP / (TP + FN)
- **F1 Score** — harmonic mean of precision and recall

***

## Authors

**Andrej Veličkov** & **Petar Ljubisavljević**  
ETF Belgrade — Neural Networks / Machine Learning course project
