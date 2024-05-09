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
    FFT = "FFT"
    PERCEPTUAL = "PERCEPTUAL"

class DescriptorTarget(Enum):
    GENERAL = "general"
    SEM = "sem"

SUPPORTED_DIMS = [16,64,256]

class AEDescriptor:
    def __init__(self, filename,target:DescriptorTarget = DescriptorTarget.GENERAL, model_path = None):

        self.target = target
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.filename = filename
        self.model_path = model_path
        self.DownloadModel()
                   

    def DownloadModel(self):
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
        SIZE = 256
        img = np.float32(image) /  np.max(image)
        img = cv.resize(img, (SIZE, SIZE))
        img = torch.tensor(img)
        img = img.to("cuda")
        img = img.unsqueeze(0).unsqueeze(0)
        output = self.model(img).detach().cpu().numpy()
        return self.model.z.detach().cpu().numpy()[0]


def GetModelName(loss_function:LossFunction, dim) -> str:
    if dim not in SUPPORTED_DIMS:
        raise ValueError(f"Dim {dim} is not supported. Supported dims are {SUPPORTED_DIMS}")
    return f"AE_{dim}_{str(loss_function.value)}.pt"   

def ListAvailableModels():
    return [(loss_function, dim) for loss_function in LossFunction for dim in SUPPORTED_DIMS]

test_img = cv.imread("../test_img.png", cv.IMREAD_GRAYSCALE)


mode = AEDescriptor(GetModelName(LossFunction.FFT, 16))
print(mode.Eval(test_img))
print(mode.Eval(test_img))



