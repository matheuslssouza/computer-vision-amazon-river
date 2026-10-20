import os
from PIL import Image
import numpy as np

def open_image(path):
    """Abre uma imagem e a converte para um array NumPy."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32)

def save_image(img_array, image_name):
    
    path = os.path.join("./data/images/", f"{image_name}.png")
    
    """Salva um array NumPy como uma imagem."""
    img = Image.fromarray(np.uint8(img_array))
    img.save(path)