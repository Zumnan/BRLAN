from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class DatasetConfig:
    NUM_CLASSES: int = 1  
    IMAGE_SIZE: Tuple[int, int] = (192, 192)  
    MEAN: tuple = (0.485, 0.456, 0.406)
    STD:  tuple = (0.229, 0.224, 0.225)
    CHANNELS: int = 1  
    BACKGROUND_CLS_ID: int = 0
    URL: str = r"https://www.kaggle.com/datasets"
    DATASET_PATH: str = os.path.join(os.getcwd(), "preprocessed_data1")
 
@dataclass(frozen=True)
class Paths:
    DATA_TRAIN_IMAGES: str = os.path.join(DatasetConfig.DATASET_PATH, "train", "images", r"*.jpg")
    DATA_TRAIN_LABELS: str = os.path.join(DatasetConfig.DATASET_PATH, "train", "masks",  r"*.jpg")
    DATA_VALID_IMAGES: str = os.path.join(DatasetConfig.DATASET_PATH, "valid", "images", r"*.jpg")
    DATA_VALID_LABELS: str = os.path.join(DatasetConfig.DATASET_PATH, "valid", "masks",  r"*.jpg")
         
@dataclass
class TrainingConfig:
    BATCH_SIZE:      int = 4 
    NUM_EPOCHS:      int = 50
    INIT_LR:       float = 3e-4
    NUM_WORKERS:     int = 0 if platform.system() == "Windows" else 12 
 
    OPTIMIZER_NAME:  str = "AdamW"
    WEIGHT_DECAY:  float = 1e-4
    USE_SCHEDULER:  bool = True 
    SCHEDULER:       str = "MultiStepLR" 
    MODEL_NAME:      str = "BRLAN"
    
     
 
@dataclass
class InferenceConfig:
    BATCH_SIZE:  int = 4
    NUM_BATCHES: int = 1
    
    
    