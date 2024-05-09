from AEDescriptor import *


img = cv.imread("test_img.png", cv.IMREAD_GRAYSCALE)
descriptor = AEDescriptor(GetModelName(LossFunction.FFT, 16))
print(descriptor.Eval(img))