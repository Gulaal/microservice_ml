import numpy as np
import torch
from skimage.transform import resize

def preprocess(pixels):
    img_32 = np.array(pixels, dtype=np.float32).reshape(32, 32)

    img_28 = resize(img_32, (28, 28), anti_aliasing=True)

    img_inv = 1.0 - (img_28 / 255.0)
    
    tensor = torch.FloatTensor(img_inv).unsqueeze(0).unsqueeze(0)
    return tensor