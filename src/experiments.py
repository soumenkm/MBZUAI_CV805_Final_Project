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

    def run(self, name: str, eviction: str | None, slots: int | None, true_labels: bool = False,
               tau: float | None = None, log_every: int = 100) -> None:
        """it runs one method over the stream: it predicts each image first, then updates the cache.
        eviction None means zero-shot, the cache is never updated. the result is saved as a json file
        and the run is skipped if that file already exists. at the end of every segment it measures 
        the heldout accuracy of every domain."""
        tau = self.tau if tau is None else tau
        run_path = Path(self.output_dir, f"{name}_{self.order_tag}_seed{self.seed}_frac{self.frac}.json")
        if run_path.exists():
            print(f"Run already exists at {run_path}")
            return

        model = CacheModel(eviction=eviction or "entropy", slots=slots, alpha=self.alpha, beta=self.beta,
                           tau=tau, true_labels=true_labels, seed=self.seed)
        seg_ends = {seg["end"]: i for i, seg in enumerate(self.segments)}
        correct, events, heldout, composition = [], [], [], []

        start_time = time.time()
        for step, idx in enumerate(tqdm(self.stream, f"Running {name} ({self.order_tag}, seed {self.seed})", unit="images")):
            correct.append(int(model.predict([idx]).item() == model.labels[idx].item()))
            if eviction is not None:
                event = model.update(idx, step)
                # only the steps where the cache changed are kept, they are enough to rebuild the cache later
                if event["added"] is not None:
                    events.append([step, event["added"], event["removed"]])

            if (step + 1) % log_every == 0:
                composition.append({"step": step + 1, "counts": self._cache_composition(model)})
            if step + 1 in seg_ends:
                heldout.append({"segment": seg_ends[step + 1], "acc": self._heldout_acc(model)})
        seconds_per_image = (time.time() - start_time) / len(self.stream)

        result = {
            "name": name,
            "eviction": eviction,
            "slots": slots,
            "cache_slots": model.slots,
            "num_anchors": model.num_anchors,
            "memory_used": model.memory_used(),
            "true_labels": true_labels,
            "alpha": self.alpha,
            "beta": self.beta,
            "tau": tau,
            "order": self.order,
            "seed": self.seed,
            "frac": self.frac,
            "segments": self.segments,
            "correct": correct,
            "events": events,
            "seconds_per_image": seconds_per_image,
            "heldout": heldout,
            "composition": composition,
        }
        self._save_json(run_path, result)
        print(f"{name}: online accuracy {sum(correct) / len(correct) * 100:.2f}%")

    def _heldout_acc(self, model: CacheModel) -> dict:
        """accuracy of the current cache on the heldout images of every domain"""
        acc = {}
        for domain, ids in self.heldout.items():
            preds = model.predict(ids)
            acc[domain] = (preds == model.labels[ids]).float().mean().item()

        return acc

    @staticmethod
    def _cache_composition(model: CacheModel) -> dict:
        """how many cached images come from each domain"""
        counts = {}
        for idx in model.cache["ids"]:
            domain = model.domains[idx]
            counts[domain] = counts.get(domain, 0) + 1

        return counts

def main(order: list, seeds: list, frac: float, runs: list) -> None:
    for seed in seeds:
        exp = Experiments(order=order, seed=seed, frac=frac)
        if seed == seeds[0]:
            exp.zero_shot_stats(taus=[0.1, 0.2, 0.3, 0.4, 0.5])
        for run in runs:
            exp.run(**run)

if __name__ == "__main__":
    runs = [
        {"name": "zero_shot", "eviction": None, "slots": None},
        {"name": "unbounded", "eviction": "entropy", "slots": None},
        {"name": "fifo", "eviction": "fifo", "slots": 3},
        {"name": "entropy", "eviction": "entropy", "slots": 3},
        {"name": "oracle", "eviction": "oracle", "slots": 3},
        {"name": "cmr", "eviction": "cmr", "slots": 3},
        # upper bound: every image enters the cache with its true label
        {"name": "true_labels", "eviction": "fifo", "slots": 3, "true_labels": True, "tau": 1.01},
    ]
    seeds = [0, 1, 2]
    main(["Art", "Clipart", "Product", "Real World"], seeds, 0.5, runs)
    main(["Real World", "Product", "Clipart", "Art"], seeds, 0.5, runs)
    main(["Clipart", "Real World", "Art", "Product"], seeds, 0.5, runs)