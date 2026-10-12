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
        self.data = self._create_data()

    def _load_classnames(self) -> list:
        """it reads the class names from the folders of the first domain.
        the class id of a class is its position in this sorted list"""
        domain_path = Path(self.data_path, self.domains[0])
        return sorted(i.name for i in domain_path.iterdir() if i.is_dir())

    def _create_data(self) -> list:
        """it lists every image with its class and domain. it doesn't open the images.
        the position of an item in this list is its id. the same id is used for the features later"""
        data = []
        for domain in self.domains:
            for label, classname in enumerate(self.classnames):
                class_path = Path(self.data_path, domain, classname)
                for image_path in sorted(class_path.iterdir()):
                    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                        continue
                    item = {
                        "id": len(data),
                        "image_path": str(image_path),
                        "label": label,
                        "classname": classname,
                        "domain": domain
                    }
                    data.append(item)

        return data

    @staticmethod
    def _load_image(image_path: str) -> Image:
        image_path = Path(image_path)
        with Image.open(image_path) as image:
            image = image.convert("RGB")

        return image

def main(seed: int) -> None:
    ds = OfficeHomeDataset(seed=seed)
    print(ds.classnames)
    print(ds[0])

if __name__ == "__main__":
    main(0)