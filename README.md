
# Drone Segmentation and Landing Zone Detection

**Drone Segmentation and Landing Zone Detection** is a deep learning-based computer vision project that identifies safe landing areas for drones from aerial images. It includes two semantic segmentation models (UNet and ResNet) and a fully interactive Streamlit dashboard for real-time visualization and testing.

This repository contains three main components:
1. UNet-based semantic segmentation model for drone imagery.
2. ResNet-based semantic segmentation model for drone imagery.
3. Streamlit dashboard for visualizing and interacting with the models.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [UNet Model](#unet-model)
  - [ResNet Model](#resnet-model)
  - [Streamlit Dashboard](#streamlit-dashboard)
- [Configuration](#configuration)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

---

## Project Structure

```
.
├── models/
│   ├── unet_segmentation.py        # UNet segmentation model code
│   ├── resnet_segmentation.py      # ResNet segmentation model code
│   └── saved_models/               # Directory to save trained models
│       ├── Unet-Boss.pt
│       └── ResNet-Bigboss.pt
├── dashboard/
│   └── streamlit_dashboard.py      # Streamlit dashboard code
├── data/
│   ├── original_images/            # Directory for original images
│   └── label_images_semantic/      # Directory for label images
├── .streamlit/
│   └── config.toml                 # Configuration file
└── README.md                       # Project README file
```

---


## Setup

### Prerequisites

- Python 3.7+
- `pip` package manager

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/drone-segmentation.git
   cd drone-segmentation
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

---


## Usage

### UNet Model

1. **Train the UNet Model**

   The UNet model code is in `models/unet_segmentation.py`. You can run this script to train the UNet model on your dataset.

   ```bash
   python models/unet_segmentation.py
   ```

2. **Evaluate the UNet Model**

   The script will automatically evaluate the model on the validation set and save the trained model as `models/saved_models/Unet-Boss.pt`.

---

### ResNet Model

1. **Train the ResNet Model**

   The ResNet model code is in `models/resnet_segmentation.py`. You can run this script to train the ResNet model on your dataset.

   ```bash
   python models/resnet_segmentation.py
   ```

2. **Evaluate the ResNet Model**

   The script will automatically evaluate the model on the validation set and save the trained model as `models/saved_models/ResNet-Bigboss.pt`.

---

### Streamlit Dashboard

1. **Run the Streamlit Dashboard**

   The Streamlit dashboard code is in `dashboard/streamlit_dashboard.py`. You can run this script to launch an interactive dashboard for visualizing model predictions and landing zone detection.

   ```bash
   streamlit run dashboard/streamlit_dashboard.py
   ```

2. **Interact with the Dashboard**

   Open your browser and go to `http://localhost:8501` to interact with the dashboard. You can upload images, visualize segmentation results, and detect safe landing zones.

---

## Configuration

The `config.toml` file is used to configure various aspects of the project, such as paths to data directories, model parameters, and Streamlit settings. Below is an example of what the `config.toml` file might look like:

```toml
[general]
project_name = "Drone Segmentation and Landing Zone Detection"

[paths]
image_dir = "/path/to/original_images"
mask_dir = "/path/to/label_images_semantic"

[model]
unet_model_path = "models/saved_models/Unet-Boss.pt"
resnet_model_path = "models/saved_models/ResNet-Bigboss.pt"

[streamlit]
title = "Drone Segmentation Dashboard"
```

---

## Results

### UNet Model

- The UNet model is trained for 15 epochs.
- The training and validation losses and accuracies are plotted and displayed.

### ResNet Model

- The ResNet model is trained for 15 epochs.
- The training and validation losses and accuracies are plotted and displayed.

---

### Visualizations

- The Streamlit dashboard allows you to visualize the input images, true masks, predicted masks, and identified safe landing zones.

---

## Acknowledgements

- The **U-Net model** was implemented from scratch based on the original architecture proposed by Olaf Ronneberger, Philipp Fischer, and Thomas Brox in *"U-Net: Convolutional Networks for Biomedical Image Segmentation"*.
- The ResNet model uses a pretrained ResNet-50 backbone for feature extraction.
- The dataset used for training and evaluation is the [**Aerial Semantic Segmentation Drone Dataset**](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset), a publicly available dataset containing aerial images and segmentation masks for drone-based scene understanding.

---

---

## Project Report

For detailed methodology, architecture, training, and results, see the full report:

[Download Project Report (PDF)](Drone_Project_Report.pdf)

---

