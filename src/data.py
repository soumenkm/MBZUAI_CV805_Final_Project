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
        self.splits = self._create_splits()

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

    def _create_splits(self) -> dict:
        """it splits each domain into visit1, visit2 and heldout images.
        the split is done class by class so that every class appears in all three parts"""
        rng = random.Random(self.seed)
        groups = {}
        for item in self.data:
            groups.setdefault((item["domain"], item["label"]), []).append(item["id"])

        splits = {domain: {"visit1": [], "visit2": [], "heldout": []} for domain in self.domains}
        for (domain, _), ids in groups.items():
            rng.shuffle(ids)
            n1 = round(len(ids) * self.split_frac[0])
            n2 = round(len(ids) * self.split_frac[1])
            splits[domain]["visit1"] += ids[:n1]
            splits[domain]["visit2"] += ids[n1:n1 + n2]
            splits[domain]["heldout"] += ids[n1 + n2:]

        return splits


    @staticmethod
    def _load_image(image_path: str) -> Image:
        image_path = Path(image_path)
        with Image.open(image_path) as image:
            image = image.convert("RGB")

        return image

def main(seed: int) -> None:
    ds = OfficeHomeDataset(seed=seed)
    print(ds.classnames)
    print(ds.data[0])
    print(ds.splits.keys())

if __name__ == "__main__":
    main(0)