import numpy as np
from core.models import PatchMetrics

class FeatureExtractor:
    """Extrai as características de uma fatia da imagem."""
    @staticmethod
    def extract_features(image_array: np.ndarray, image_id: int, x: int, y: int) -> PatchMetrics:
        mean_r = float(np.mean(image_array[:, :, 0]))
        mean_g = float(np.mean(image_array[:, :, 1]))
        mean_b = float(np.mean(image_array[:, :, 2]))
        
        gray = (
            0.299 * image_array[:, :, 0]
            + 0.587 * image_array[:, :, 1]
            + 0.114 * image_array[:, :, 2]
        )
        
        std_dev = float(np.std(gray))
        grad_h = np.mean(np.abs(gray[:, :-1] - gray[:, 1:]))
        grad_v = np.mean(np.abs(gray[:-1, :] - gray[1:, :]))
        gradient = float((grad_h + grad_v) / 2.0)
        
        return PatchMetrics(image_id, x, y, f"bloco_{image_id:04d}", mean_r, mean_g, mean_b, std_dev, gradient)
