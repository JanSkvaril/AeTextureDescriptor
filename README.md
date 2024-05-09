# AE-based Texture Descriptor
Autoencoder-based texture descriptor proposed in Master Thesis [TODO:cite]. The autoencoders were trained with different texture-oriented loss functions and latent space dimmensions. It can be used for the same purposes as any other descriptor.

## Targets
*Target* defines what was the training dataset. Two targets are currently supported:
- General texture - trained with general texture dataset, as described in the thesis
- SEM texture - trained with SEM texture images

## Exmple usage
Install after clonning:
```
pip install .
```

Usage:
```python
from AEDescriptor import * 
imgage = cv.imread("image.png", cv.IMREAD_GRAYSCALE)
mode = AEDescriptor(GetModelName(loss_function=LossFunction.FFT, dim=16))
mode.Eval(image)
# => [0.67124015 0.6368097 ... 0.41859195] 16 features
```
To get list of all currently avaiable descriptors:
```python
print(ListAvailableModels())

```

