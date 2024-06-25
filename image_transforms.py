# image_transforms.py

import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

def normalize(tensor):
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)

def histogram_equalization(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    
    # Perform histogram equalization
    image_yuv = cv2.cvtColor(image_np, cv2.COLOR_RGB2YUV)
    image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
    image_eq = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
    
    # Convert back to PIL Image
    return Image.fromarray(image_eq)

def contrast_enhancement(image):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(1.5)  # Increase contrast by 50%

def get_preprocessing_transforms(config): # TODO: break down this function into two functions: get_pil_preprocessing_transforms, and get_tensor_preprocessing_transforms and modify get_transforms accordingly
    preprocessing_steps = []
    if config['preprocessing']['histogram_equalization']:
        preprocessing_steps.append(histogram_equalization)
    if config['preprocessing']['contrast_enhancement']:
        preprocessing_steps.append(contrast_enhancement)

    def preprocess(image):
        for step in preprocessing_steps:
            image = step(image)
        return image

    return preprocess

def get_augmentation_transforms(config):
    augmentations = []
    if config['augmentation']['rotation']:
        augmentations.append(transforms.RandomRotation(15))
    if config['augmentation']['flip']:
        augmentations.append(transforms.RandomHorizontalFlip())
    return augmentations

def get_transforms(config):
    augmentations = get_augmentation_transforms(config)
    preprocess_transforms = get_preprocessing_transforms(config)

    # PIL image transforms (including custom preprocessing)
    pil_transforms = [
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: preprocess_transforms(x))
    ]

    # Tensor transforms
    tensor_transforms = [
        transforms.ToTensor()
    ]

    # Add normalization if enabled in config
    if config['preprocessing']['normalize']:
        tensor_transforms.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    
    # Combine all transforms
    train_transform = transforms.Compose(augmentations + pil_transforms + tensor_transforms)
    test_transform = transforms.Compose(pil_transforms + tensor_transforms)

    return train_transform, test_transform