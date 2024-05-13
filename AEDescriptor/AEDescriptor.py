"""
Autoencoder based descriptor for texture analysis.
The descriptor is trained on a dataset of textures and can be used to extract features from images.
author: Jan Skvaril
LICENCE: MIT

"""


import torch
from enum import Enum
import json
import tempfile
import os
import ml
import numpy as np
import cv2 as cv


MODEL_REPOSITORY_URL = "https://github.com/JanSkvaril/AeTextureDescriptor/raw/main/models"

class LossFunction(Enum):
    """Loss function used for training the autoencoder"""
    FFT = "FFT"
    PERCEPTUAL = "Perceptural"
    AUTOCORRELATION = "AutoCorr"

class DescriptorTarget(Enum):
    """Target of the descriptor"""

    # General target, trained on a general texture dataset
    GENERAL = "general"
    # Target trained on a dataset of SEM images
    SEM = "sem"

# Supported dimensions of the descriptor
SUPPORTED_DIMS = [16,64,256]

class AEDescriptor:
    def __init__(self, filename = "",target:DescriptorTarget = DescriptorTarget.GENERAL, model_path = None):
        """
            Autoencoder based descriptor for texture analysis.
            - filename: Name of the model file to load, use GetModelName to get the name. If empty, default model is loaded
            - target: Target of the descriptor (General or SEM)
            - model_path: Path to the directory where the models are stored. If None, the models are stored in a temporary directory
            The model is downloaded from the repository if it is not present in the model_path

            Example usage:
            ```
            imgage = cv.imread("image.png", cv.IMREAD_GRAYSCALE)
            mode = AEDescriptor(GetModelName(LossFunction.FFT, 16))
            mode.Eval(image)
            # => [0.67124015 0.6368097 ... 0.41859195] 16 features
            ```
        """
        if filename == "": 
            filename = GetModelName(LossFunction.PERCEPTUAL, 16)
        self.target = target
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.filename = filename
        self.model_path = model_path
        self.DownloadModel()
                   

    def DownloadModel(self):
        """Download the model from the repository if it is not present in the model_path"""
        cache_dir = tempfile.gettempdir()
        if self.model_path is not None:
            cache_dir = self.model_path
        cache_dir = os.path.join(cache_dir,self.target.value)
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        destination = f"{cache_dir}/{self.filename}"

        if not os.path.exists(destination):     
            print(f"Downloading model to {cache_dir}")
            url = f"{MODEL_REPOSITORY_URL}/{self.target.value}/{self.filename}"      
            torch.hub.download_url_to_file(url, dst=destination)
        model = torch.load(destination, map_location="cpu")
        model.eval()
        model.to(self.device)
        self.model = model
        return model
    
    def Eval(self, image : np.ndarray):
        """Evaluate the descriptor on the image"""
        SIZE = 256
        img = np.float32(image) /  np.max(image)
        img = cv.resize(img, (SIZE, SIZE))
        img = torch.tensor(img)
        img = img.to(self.device)
        img = img.unsqueeze(0).unsqueeze(0)
        output = self.model(img).detach().cpu().numpy()
        return self.model.z.detach().cpu().numpy()[0]


def GetModelName(loss_function:LossFunction, dim) -> str:
    """Get the name of the model file"""
    if dim not in SUPPORTED_DIMS:
        raise ValueError(f"Dim {dim} is not supported. Supported dims are {SUPPORTED_DIMS}")
    return f"AE_{dim}_{str(loss_function.value)}.pt"   

def ListAvailableModels():
    """List all available models"""
    return [(loss_function, dim) for loss_function in LossFunction for dim in SUPPORTED_DIMS]




