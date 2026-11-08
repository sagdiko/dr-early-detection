from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class FundusDataset(Dataset):
    """CSV-backed dataset; CSV columns: image_path,label,patient_id."""
    def __init__(self, csv_path, transform=None):
        self.frame = pd.read_csv(csv_path)
        self.transform = transform
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = Image.open(Path(row.image_path)).convert("RGB")
        if self.transform:
            image = self.transform(image=image)["image"]
        return {"image": image, "label": int(row.label), "patient_id": str(row.patient_id)}
