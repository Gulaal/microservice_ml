import numpy as np
import torch
from skimage.transform import resize

def preprocess(pixels):
    img = np.array(pixels, dtype=np.float32).reshape(28, 28)
    
    img = img / 255.0
    
    binary = img > 0.1
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        cropped = img[rmin:rmax+1, cmin:cmax+1]
        
        h, w = cropped.shape
        if h > w:
            new_h = 20
            new_w = int(20 * w / h)
        else:
            new_w = 20
            new_h = int(20 * h / w)
        resized = resize(cropped, (new_h, new_w), anti_aliasing=True)
        
        final = np.zeros((28, 28), dtype=np.float32)
        y_off = (28 - new_h) // 2
        x_off = (28 - new_w) // 2
        final[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    else:
        final = img

    final = (final - 0.1307) / 0.3081
    
    return torch.FloatTensor(final).unsqueeze(0).unsqueeze(0)