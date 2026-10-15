import json
import time
import torch
from pathlib import Path
from tqdm import tqdm
from data import OfficeHomeDataset
from models import CacheModel

class Experiments:
    def __init__(self, order: list, seed: int, frac: float = 1.0, alpha: float = 2.0, beta: float = 5.0, tau: float = 0.3) -> None:
        self.order = order
        self.seed = seed
        self.frac = frac
        self.alpha = alpha
        self.beta = beta
        self.tau = tau

        self.cwd = Path.cwd()
        self.output_dir = Path(self.cwd, "outputs/runs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = OfficeHomeDataset(seed=seed)
        self.stream, self.segments = self.dataset.create_stream(order, frac=frac)
        self.heldout = {domain: self.dataset.splits[domain]["heldout"] for domain in self.dataset.domains}
        self.order_tag = "".join(domain[0] for domain in order)

    def _load_json(self, json_path: str) -> dict:
        """it loads a saved json file"""
        json_path = Path(json_path)
        with open(json_path, mode="r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"JSON file loaded from {json_path}")

        return data

    def _save_json(self, json_path: str, data: dict) -> None:
        """it saves the results in a json file. it writes a temp file first so a crash never leaves a broken file"""
        json_path = Path(json_path)
        temp_path = json_path.with_suffix(".temp")
        with open(temp_path, mode="w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        temp_path.replace(json_path)
        print(f"JSON file saved at {json_path}")

    def zero_shot_stats(self, taus: list) -> None:
        """it saves the zero-shot accuracy of every domain and, for every tau, how many images
        would enter the cache and how many of them have a correct label"""
        model = CacheModel()
        _, preds, entropy, _ = model.zero_shot(model.image_feats)
        correct = (preds == model.labels).float()

        stats = {}
        for domain in self.dataset.domains:
            mask = torch.tensor([d == domain for d in model.domains], device=model.device)
            stats[domain] = {"zero_shot_acc": correct[mask].mean().item()}
            for tau in taus:
                admitted = mask & (entropy < tau)
                stats[domain][f"tau{tau}"] = {
                    "admitted": (admitted.sum() / mask.sum()).item(),
                    "label_acc": correct[admitted].mean().item()
                }

        self._save_json(Path(self.cwd, "outputs/zero_shot_stats.json"), stats)

def main(order: list, seed: int, frac: float) -> None:
    exp = Experiments(order=order, seed=seed, frac=frac)
    exp.zero_shot_stats(taus=[0.1, 0.2, 0.3, 0.4, 0.5])

if __name__ == "__main__":
    main(["Art", "Clipart", "Product", "Real World"], 0, 0.1)