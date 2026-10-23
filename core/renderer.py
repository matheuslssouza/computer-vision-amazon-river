import numpy as np
import pandas as pd
from PIL import Image

class MapRenderer:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.color_palette = {
            "rio": np.array([0, 0, 255]),  # Azul
            "mata": np.array([0, 255, 0]),  # Verde
        }
       
    
    ### Renderiza o mapa pintando com base nas previsões do modelo de aprendizado de máquina já estabelecidas na coluna label_predict da tabela excel.
    def render(self, df: pd.DataFrame, original_image: np.ndarray, window_size: int = 16) -> Image.Image:
        
        map_colored = original_image.copy()
        
        for index, data in df.iterrows():
            x = data["coordinate_x"]
            y = data["coordinate_y"]
            pred = str(data["label_predict"]).strip().lower()
            
            if pred in self.color_palette:
                color = self.color_palette[pred]
                patch_original = map_colored[y:y+window_size, x:x+window_size]
                map_colored[y:y+window_size, x:x+window_size] = self.alpha * color + (1 - self.alpha) * patch_original
        
        return Image.fromarray(np.uint8(map_colored))