
from dataclasses import asdict, dataclass

@dataclass
class PatchMetrics:
    id: int
    coordinate_x: int
    coordinate_y: int
    archive: str
    mean_r: float
    mean_g: float
    mean_b: float
    std_dev: float
    gradient: float
    label: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)
    