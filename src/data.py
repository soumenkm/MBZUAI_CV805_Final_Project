import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from config import domains, split_frac
import math

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

    def __len__(self) -> int:
            return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        if idx >= len(self):
            raise IndexError("Index is out of range!")

        return self.data[idx]

    def create_stream(self, order: list, frac: float = 1.0) -> tuple:
        """it creates the test stream for a domain order. every domain is visited twice,
        with different images each time. it returns the image ids in stream order and
        the segments, where each segment is one visit of one domain"""
        rng = random.Random(self.seed)
        stream = []
        segments = []
        for visit in ["visit1", "visit2"]:
            for domain in order:
                ids = self.splits[domain][visit].copy()
                # without the shuffle the images would come class by class
                rng.shuffle(ids)
                ids = ids[:math.floor(len(ids) * frac)]
                segments.append({"domain": domain, "visit": visit, "start": len(stream), "end": len(stream) + len(ids)})
                stream += ids

        return stream, segments
    
    def prompt_classnames(self) -> list:
        """it returns the class names in a form usable in the text prompt, e.g. Alarm_Clock -> alarm clock"""
        return [c.replace("_", " ").lower() for c in self.classnames]

    @staticmethod
    def _load_image(image_path: str) -> Image:
        image_path = Path(image_path)
        with Image.open(image_path) as image:
            image = image.convert("RGB")

        return image

def main(seed: int, order: list) -> None:
    ds = OfficeHomeDataset(seed=seed)
    print(ds.classnames)
    print(ds.data[0])
    print(ds.splits.keys())

    print(len(ds))
    print(ds[0])

    for domain in ds.domains:
        print(domain, {k: len(v) for k, v in ds.splits[domain].items()})

    stream, segments = ds.create_stream(order)
    print(len(stream))
    for seg in segments:
        print(seg)

    print([ds[i]["label"] for i in stream[:20]])   # labels should look mixed, not 0,0,0,...
    s = ds.splits["Art"]
    print(len(set(s["visit1"]) & set(s["visit2"])), len(set(s["visit1"]) & set(s["heldout"])))   # both should be 0
    

if __name__ == "__main__":
   main(0, ["Art", "Clipart", "Product", "Real World"])