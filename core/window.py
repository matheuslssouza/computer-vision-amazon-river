import numpy as np
import os
from PIL import Image

class SlidingWindow:
    def __init__(self, window_size: int = 16, save_dir: str = "./data/slices"):
        self.window_size = window_size
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    
    def slice_image(self, image_array: np.ndarray) -> Iterator[Tuple[int,int,int,np.ndarray]]:
        height, width, channels = image_array.shape
        patch_id = 0
        
        for y in range(0, height, self.window_size):
            for x in range(0, width, self.window_size):
                
                patch = image_array[y:y+self.window_size, x:x+self.window_size]
                
                if patch.shape[0] < self.window_size or patch.shape[1] < self.window_size:
                    continue
                
                patch_id += 1
                
                if self.save_dir:
                    archive_path = os.path.join(self.save_dir, f"patch_{patch_id:04d}.png")
                      
                patch_image = Image.fromarray(np.uint8(patch))
                patch_image.save(archive_path)
                                             
                yield(patch, patch_id, x, y)
        
    