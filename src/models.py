import torch
import torch.nn.functional as F
import open_clip
from pathlib import Path
from tqdm import tqdm
from data import OfficeHomeDataset
from config import clip_name, template
import math
import random

class CacheModel:
    def __init__(self, eviction: str = "entropy", slots: int | None = 3, alpha: float = 2.0, beta: float = 5.0, tau: float = 0.3, seed: int = 0, true_labels: bool = False) -> None:
        if eviction not in ["fifo", "random", "entropy", "oracle"]:
            raise ValueError(f"Unknown eviction rule: {eviction}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cwd = Path.cwd()
        self.weights_path = Path(self.cwd, "models/vit_base_patch16_clip_224.openai/open_clip_pytorch_model.bin")
        self.feature_path = Path(self.cwd, "data/features_vitb16.pt")

        # the saved features are loaded. on the very first run they don't, so extract_features must run first
        if self.feature_path.exists():
            feats = torch.load(self.feature_path)
            self.image_feats = feats["image_feats"].to(self.device)
            self.text_feats = feats["text_feats"].to(self.device)
            self.labels = feats["labels"].to(self.device)
            self.domains = feats["domains"]
            self.num_classes = len(self.text_feats)

        self.alpha = alpha
        self.beta = beta
        self.reset()

        self.eviction = eviction
        self.slots = slots
        self.tau = tau
        self.true_labels = true_labels
        self.rng = random.Random(seed)

    def extract_features(self, dataset: OfficeHomeDataset, batch_size: int = 256) -> None:
        """it runs frozen clip once over all images and class prompts and saves the normalized features.
        the row i of the image features belongs to the image with id i in the dataset.
        it does nothing if the features are already saved"""
        if self.feature_path.exists():
            print(f"Features already exist at {self.feature_path}")
            return

        model, _, preprocess = open_clip.create_model_and_transforms(clip_name, pretrained=str(self.weights_path))
        tokenizer = open_clip.get_tokenizer(clip_name)
        model = model.to(self.device).eval()

        with torch.no_grad():
            prompts = tokenizer([template.format(c) for c in dataset.prompt_classnames()]).to(self.device)
            text_feats = model.encode_text(prompts).float().cpu()

            image_feats = []
            for start in tqdm(range(0, len(dataset), batch_size), "Extracting image features", unit="batch"):
                items = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
                images = torch.stack([preprocess(dataset._load_image(item["image_path"])) for item in items]).to(self.device)
                image_feats.append(model.encode_image(images).float().cpu())

        feats = {
            "image_feats": F.normalize(torch.cat(image_feats), dim=-1),
            "text_feats": F.normalize(text_feats, dim=-1),
            "labels": torch.tensor([item["label"] for item in dataset.data]),
            "domains": [item["domain"] for item in dataset.data],
            "classnames": dataset.classnames
        }
        torch.save(feats, self.feature_path)
        print(f"Features saved at {self.feature_path}")

    def zero_shot(self, feats: torch.Tensor) -> tuple:
        """it returns the zero-shot logits, the predicted class, the entropy and the top probability.
        the entropy is divided by its maximum value so it is always between 0 and 1"""
        logits = 100 * feats @ self.text_feats.T
        probs = logits.softmax(dim=-1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1) / math.log(self.num_classes)
        return logits, probs.argmax(-1), entropy, probs.max(-1).values

    def reset(self) -> None:
        """it empties the cache before a new run. the cache stores image ids,
        and their features are looked up from the saved features when needed"""
        self.cache = {"ids": [], "labels": [], "entropy": [], "time": []}

    def cache_logits(self, feats: torch.Tensor, ids: list = None, labels: list = None) -> torch.Tensor:
        """it adds up the votes of the cached images for every class.
        by default it uses the current cache, but other images can be passed to try a cache that doesn't exist"""
        ids = self.cache["ids"] if ids is None else ids
        labels = self.cache["labels"] if labels is None else labels
        if len(ids) == 0:
            return torch.zeros(len(feats), self.num_classes, device=self.device)

        votes = torch.exp(-self.beta * (1 - feats @ self.image_feats[ids].T))
        onehot = F.one_hot(torch.tensor(labels, device=self.device), self.num_classes).float()
        # each vote is added to the class of its cached image
        return self.alpha * votes @ onehot

    def predict(self, ids: list) -> torch.Tensor:
        """it predicts the class of the given images using zero-shot logits plus the cache votes"""
        feats = self.image_feats[ids]
        logits = self.zero_shot(feats)[0] + self.cache_logits(feats)
        return logits.argmax(-1)

    def update(self, idx: int, step: int) -> dict:
        """it decides if the image enters the cache and, if its class is full, which entry leaves.
        it is called after the image is predicted. it returns the image ids added and removed (None if nothing changed)"""
        _, pred, entropy, _ = self.zero_shot(self.image_feats[[idx]])
        c, entropy = pred.item(), entropy.item()
        if self.true_labels:
            c = self.labels[idx].item()
        event = {"step": step, "added": None, "removed": None}
        if entropy >= self.tau:
            return event

        in_class = [i for i, label in enumerate(self.cache["labels"]) if label == c]
        if self.slots is not None and len(in_class) >= self.slots:
            pos = self.evict(c, idx, in_class)
            event["removed"] = self.cache["ids"][pos]
            for key in self.cache:
                self.cache[key].pop(pos)

        for key, val in zip(self.cache, [idx, c, entropy, step]):
            self.cache[key].append(val)
        event["added"] = idx

        return event

    def evict(self, c: int, idx: int, in_class: list) -> int:
        """it returns the cache position of the class c entry to remove, using the eviction rule of this run.
        in_class has the cache positions of all class c entries"""
        if self.eviction == "fifo":
            return min(in_class, key=lambda i: self.cache["time"][i])
        if self.eviction == "random":
            return self.rng.choice(in_class)
        if self.eviction == "oracle":
            # the oracle can see true labels, so wrongly labelled entries go first
            wrong = [i for i in in_class if self.labels[self.cache["ids"][i]].item() != c]
            in_class = wrong if wrong else in_class

        # remove the least confident entry. this is the entropy rule, and also the fallback for oracle
        return max(in_class, key=lambda i: self.cache["entropy"][i])

def main(runs: list, tau: float) -> None:
    ds = OfficeHomeDataset(seed=0)
    stream, segments = ds.create_stream(ds.domains, frac=0.1)

    model = CacheModel(tau=tau)
    zs_correct = (model.zero_shot(model.image_feats[stream])[1] == model.labels[stream]).float()
    results = {"zero-shot": zs_correct}

    for name, eviction, slots, beta, true_labels, run_tau in runs:
        model = CacheModel(eviction=eviction, slots=slots, beta=beta, tau=run_tau, true_labels=true_labels)
        correct = []
        for step, idx in enumerate(tqdm(stream, f"Running {name}", unit="images")):
            # predict first, then update, so an image never helps its own prediction
            correct.append(int(model.predict([idx]).item() == model.labels[idx].item()))
            model.update(idx, step)
        results[name] = torch.tensor(correct, dtype=torch.float)
        print(f"{name}: final cache size {len(model.cache['ids'])}")

    print("\nonline accuracy (%) per segment")
    print(f"{'method':<12}{'all':>7}" + "".join(f"{s['domain'][:4] + s['visit'][-1]:>8}" for s in segments))
    for name, correct in results.items():
        correct = correct.cpu()
        row = "".join(f"{correct[s['start']:s['end']].mean().item() * 100:>8.2f}" for s in segments)
        print(f"{name:<12}{correct.mean().item() * 100:>7.2f}{row}")

if __name__ == "__main__":
    main(runs=[("beta5", "entropy", 3, 5.0, False, 0.3), ("beta10", "entropy", 3, 10.0, False, 0.3),
               ("true5_all", "fifo", 3, 5.0, True, 1.01), ("true10_all", "fifo", 3, 10.0, True, 1.01)], tau=0.3)