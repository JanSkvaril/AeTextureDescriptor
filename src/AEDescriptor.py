import torch
from enum import Enum
import json

class AeDescriptorLoss(Enum):
    FFT = "FFT"
    Perceptual = "Perceptual"

class AEDescriptor:
    def __init__(self, dim, loss : AeDescriptorLoss, config_path = None, model_path = None):
        self.dim = dim
        self.loss = loss
        self.config_path = config_path  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.GetModelNameFromConfig(config_path, dim, loss)
   
                   
    def GetModelNameFromConfig(self,config_path):
        filename = None
        with open(config_path) as f:
            self.config = json.load(f)
            models = self.config["general"]
            for model_config in models:
                if model_config["dim"] == self.dim and model_config["loss"] == self.loss.value:
                    filename = model_config["filename"]
        if filename is None:
            raise Exception("Model not found")
        self._model_filename = filename
        return filename

    def DownloadModel(self):
        pass

AEDescriptor(64, AeDescriptorLoss.FFT, "model_config.json")

