import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_transforms(image_size=300, is_train=True):
    """Return the appropriate image transforms for training or validation."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class RetinalDataset(Dataset):
    """
    APTOS 2019 retinal fundus image dataset.

    Applies CLAHE contrast enhancement before standard ImageNet normalisation.
    Each sample returns (image_tensor, label) where label is 0–4 (DR grade).
    """

    GRADE_NAMES = {
        0: "No DR",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative DR",
    }

    def __init__(self, df, transform=None):
        """
        Args:
            df: pandas DataFrame with columns ['filepath', 'diagnosis'].
            transform: torchvision transforms to apply after CLAHE.
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _apply_clahe(self, img_rgb: np.ndarray) -> np.ndarray:
        """Apply CLAHE to the L channel of an LAB image."""
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["filepath"])
        if img is None:
            raise FileNotFoundError(f"Image not found: {row['filepath']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self._apply_clahe(img)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, int(row["diagnosis"])