#!/Library/Frameworks/Python.framework/Versions/3.11/bin/python3
from sklearn.discriminant_analysis import StandardScaler
import streamlit as st
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import radiomics
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import sys
import pandas as pd
# import albumentations as A
import cv2
import torchvision.transforms as T
import torchvision.models as models
import matplotlib.pyplot as plt

print(sys.executable)

# Load models
segmentation_model_path ='/Users/mustafasoydan/Desktop/projects/computational_buse/models/ResNet-Bigboss (1).pt'

# Define Models
class ResNetSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(ResNetSegmentation, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # Remove the last two layers
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        x = self.resnet(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x
    
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Instantiate the model
num_classes = 24  # Number of segmentation classes
#model = ResNetSegmentation(num_classes).to(device)

# Custom Dataset class
class DroneDataset(Dataset):
    def __init__(self, image_dir, mask_dir, files, mean, std, transform=None, patch=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.files = files
        self.transform = transform
        self.patches = patch
        self.mean = mean
        self.std = std
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Retrieve the image and mask file names
        img_file = self.files[idx]
        base_name = os.path.splitext(img_file)[0]
        
        # Build the file paths for the image and mask using the provided index
        img_path = os.path.join(self.image_dir, img_file)            # Path to the input image
        mask_path = os.path.join(self.mask_dir, base_name + '.png')  # Path to the corresponding segmentation mask

        # Read the image and mask
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image file not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']
        
        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)
        mask = torch.from_numpy(mask).long()
        
        if self.patches:
            img, mask = self.tiles(img, mask)
            
        return img, mask
    
    def tiles(self, img, mask):
        img_patches = img.unfold(1, 512, 512).unfold(2, 768, 768)
        img_patches = img_patches.contiguous().view(3, -1, 512, 768)
        img_patches = img_patches.permute(1, 0, 2, 3)
        
        mask_patches = mask.unfold(0, 512, 512).unfold(1, 768, 768)
        mask_patches = mask_patches.contiguous().view(-1, 512, 768)
        
        return img_patches, mask_patches


def dashboard():
    st.title("Segmentation Dashboard")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "dcm"])

    if uploaded_file is not None:
        
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        original_image = image.save("original.png")
        
        st.write("Performing Segmentation...")
        
        # Define mean and std
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.Compose([
            transforms.Resize((704, 1056), interpolation=cv2.INTER_NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        image_tensor = transform(image).unsqueeze(0)
        print('transfomraiton done') 
        segmentation_model = ResNetSegmentation(num_classes).to(device)
        segmentation_model.load_state_dict(torch.load(segmentation_model_path, map_location=torch.device('cpu')))
        segmentation_model.eval()
        with torch.no_grad():
            mask = segmentation_model(image_tensor)
            mask = torch.sigmoid(mask)
        print('segmentation done')

        _, preds = torch.max(mask, dim=1)

        # Move predictions and masks to CPU for analysis
        preds = preds.cpu().numpy()
        # Define class indices mapping (if necessary)

        class_mapping = {
            0: "unlabeled",
            1: "paved-area",
            2: "dirt",
            3: "grass",
            4: "gravel",
            5: "water",
            6: "rocks",
            7: "pool",
            8: "vegetation",
            9: "roof",
            10: "wall",
            11: "window",
            12: "door",
            13: "fence",
            14: "fence-pole",
            15: "person",
            16: "dog",
            17: "car",
            18: "bicycle",
            19: "tree",
            20: "bald-tree",
            21: "ar-marker",
            22: "obstacle",
            23: "conflicting"
        }

        # Check predicted class indices and corresponding class names:
        unique_pred_classes = np.unique(preds)
        print("Predicted Classes:")
        for cls in unique_pred_classes:
            print(f"Class Index: {cls}, Class Name: {class_mapping.get(cls, 'Unknown')}")
        

        # Visualization function
        def visualize_predictions(preds, class_mapping):
            """
            Visualize predictions alongside the input images.

            :param image_tensor: Tensor of the input image (with shape [1, C, H, W])
            :param preds: Numpy array of predictions
            :param class_mapping: Dictionary mapping class indices to class names
            """

            plt.figure(figsize=(30, 10))
            plt.subplot(1, 2, 2)
            plt.title("Predicted Mask")
            plt.imshow(preds[0], cmap='tab20')  # Display the predicted mask with colormap
            plt.colorbar(ticks=range(len(class_mapping)))  # Add colorbar with class ticks

            st.pyplot(plt)  # Use Streamlit's pyplot function to display the plot

        # After obtaining predictions, call the visualize function
        visualize_predictions(preds, class_mapping)

dashboard()