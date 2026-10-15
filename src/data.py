import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from config import domains, split_frac

class OfficeHomeDataset(Dataset):
    def __init__(self, seed: int = 0) -> None:
        """seed controls the split of each domain and the order of images in the stream"""
        self.seed = seed
        self.domains = domains
        self.split_frac = split_frac

        self.cwd = Path.cwd()
        self.data_path = Path(self.cwd, "data/OfficeHomeDataset")

        self.classnames = self._load_classnames()

    def _load_classnames(self) -> list:
        """it reads the class names from the folders of the first domain.
        the class id of a class is its position in this sorted list"""
        domain_path = Path(self.data_path, self.domains[0])
        return sorted(i.name for i in domain_path.iterdir() if i.is_dir())

    @staticmethod
    def _load_image(image_path: str) -> Image:
        image_path = Path(image_path)
        with Image.open(image_path) as image:
            image = image.convert("RGB")

        return image

def main(seed: int) -> None:
    ds = OfficeHomeDataset(seed=seed)
    print(ds.classnames)

if __name__ == "__main__":
    main(0)