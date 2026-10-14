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
    def __init__(self, eviction: str = "entropy", slots: int | None = 3, alpha: float = 2.0, beta: float = 5.0, tau: float = 0.3, seed: int = 0, true_labels: bool = False, anchor_frac: float = 0.25,) -> None:
        if eviction not in ["fifo", "random", "entropy", "oracle", "cmr"]:
            raise ValueError(f"Unknown eviction rule: {eviction}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cwd = Path.cwd()
        self.weights_path = Path(self.cwd, "models/vit_base_patch16_clip_224.openai/open_clip_pytorch_model.bin")
        self.feature_path = Path(self.cwd, "data/features_vitb16.pt")

        # the saved features are loaded if they exist. on the very first run they don't, so extract_features must run first
        if self.feature_path.exists():
            feats = torch.load(self.feature_path)
            self.image_feats = feats["image_feats"].to(self.device)
            self.text_feats = feats["text_feats"].to(self.device)
            self.labels = feats["labels"].to(self.device)
            self.domains = feats["domains"]
            self.num_classes = len(self.text_feats)

        self.eviction = eviction
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.true_labels = true_labels
        self.rng = random.Random(seed)

        # anchors use part of the memory, so cmr gets fewer cache slots than the other methods.
        self.slots = slots
        self.num_anchors = 0
        if eviction == "cmr":
            self.num_anchors = round(anchor_frac * slots * self.num_classes)
            self.slots = (slots * self.num_classes - self.num_anchors) // self.num_classes

        self.reset()

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
        self.anchors = {"ids": [], "labels": [], "weights": []}
        self.num_seen = 0

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
        _, pred, entropy, top_prob = self.zero_shot(self.image_feats[[idx]])
        c, entropy, top_prob = pred.item(), entropy.item(), top_prob.item()
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

        # the image becomes an anchor only after the eviction, so it never protects itself
        if self.num_anchors > 0:
            self.add_anchor(idx, c, top_prob)

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

        if self.eviction == "cmr" and len(self.anchors["ids"]) > 0:
            harm = self.cmr_harm(c, idx, in_class)
            return in_class[harm.argmin().item()]

        # remove the least confident entry.
        return max(in_class, key=lambda i: self.cache["entropy"][i])

    def add_anchor(self, idx: int, label: int, weight: float) -> None:
        """it keeps a fixed size random sample of past admitted images as anchors (reservoir sampling).
        label is the zero-shot label and weight is the zero-shot top probability"""
        self.num_seen += 1
        item = [idx, label, weight]
        if len(self.anchors["ids"]) < self.num_anchors:
            for key, val in zip(self.anchors, item):
                self.anchors[key].append(val)
            return

        # every past image keeps the same chance num_anchors / num_seen of being an anchor
        pos = self.rng.randrange(self.num_seen)
        if pos < self.num_anchors:
            for key, val in zip(self.anchors, item):
                self.anchors[key][pos] = val

    def _anchor_tensors(self) -> tuple:
        """it returns the features, labels and weights of the anchors as tensors"""
        feats = self.image_feats[self.anchors["ids"]]
        labels = torch.tensor(self.anchors["labels"], device=self.device)
        weights = torch.tensor(self.anchors["weights"], device=self.device)
        return feats, labels, weights

    @staticmethod
    def _margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """score of the label minus the best other score, for every row"""
        rows = torch.arange(len(labels), device=logits.device)
        others = logits.clone()
        others[rows, labels] = -float("inf")
        return logits[rows, labels] - others.max(-1).values

    def cmr_harm_slow(self, c: int, idx: int, in_class: list) -> torch.Tensor:
        """for every candidate it rebuilds the cache with the candidate swapped for the new image
        and adds up how much worse the anchors get. simple but slow, used to check cmr_harm"""
        feats, labels, weights = self._anchor_tensors()
        zs_logits = self.zero_shot(feats)[0]
        margin_old = self._margin(zs_logits + self.cache_logits(feats), labels)

        harm = []
        for pos in in_class:
            ids = [i for k, i in enumerate(self.cache["ids"]) if k != pos] + [idx]
            labs = [l for k, l in enumerate(self.cache["labels"]) if k != pos] + [c]
            margin_new = self._margin(zs_logits + self.cache_logits(feats, ids, labs), labels)
            loss_change = F.softplus(-margin_new) - F.softplus(-margin_old)
            harm.append((weights * loss_change.clamp(min=0)).sum())

        return torch.stack(harm)

    def cmr_harm(self, c: int, idx: int, in_class: list) -> torch.Tensor:
        """same result as cmr_harm_slow, but faster. a swap only changes the class c score,
        so the new margins are computed directly without rebuilding the cache"""
        feats, labels, weights = self._anchor_tensors()
        logits = self.zero_shot(feats)[0] + self.cache_logits(feats)
        rows = torch.arange(len(labels), device=self.device)

        cand_ids = [self.cache["ids"][i] for i in in_class]
        vote_new = torch.exp(-self.beta * (1 - feats @ self.image_feats[idx]))           # (anchors,)
        vote_old = torch.exp(-self.beta * (1 - feats @ self.image_feats[cand_ids].T))    # (anchors, candidates)
        new_c = logits[:, c:c + 1] + self.alpha * (vote_new[:, None] - vote_old)         # class c score after each swap

        # best score among classes that are neither the anchor's label nor c, a swap never changes it
        others = logits.clone()
        others[rows, labels] = -float("inf")
        others[:, c] = -float("inf")
        best_other = others.max(-1).values[:, None]
        own = logits[rows, labels][:, None]

        # if the anchor's label is c the swap moves its own score, otherwise it can only move a rival
        margin_new = torch.where((labels == c)[:, None], new_c - best_other, own - torch.maximum(best_other, new_c))
        margin_old = self._margin(logits, labels)[:, None]

        loss_change = F.softplus(-margin_new) - F.softplus(-margin_old)
        return (weights[:, None] * loss_change.clamp(min=0)).sum(0)

    def memory_used(self) -> int:
        """number of stored images, cache plus anchors"""
        return len(self.cache["ids"]) + len(self.anchors["ids"])

def main(runs: list, tau: float, check_steps: int) -> None:
    ds = OfficeHomeDataset(seed=0)
    stream, segments = ds.create_stream(ds.domains, frac=0.1)

    # check 1: fast and slow cmr harm should give the same numbers
    model = CacheModel(eviction="cmr", slots=3, tau=tau)
    for step, idx in enumerate(stream[:check_steps]):
        model.update(idx, step)
    diffs, same_choice = [], 0
    classes = sorted(set(model.cache["labels"]))
    for c in classes:
        in_class = [i for i, label in enumerate(model.cache["labels"]) if label == c]
        fast = model.cmr_harm(c, stream[check_steps], in_class)
        slow = model.cmr_harm_slow(c, stream[check_steps], in_class)
        diffs.append((fast - slow).abs().max().item())
        same_choice += int(fast.argmin() == slow.argmin())
    print(f"cmr check: max difference {max(diffs):.2e}, same choice in {same_choice} of {len(classes)} classes")
    print(f"cmr memory: {model.slots} cache slots per class + {model.num_anchors} anchors, memory used {model.memory_used()}")

    # check 2: cmr against entropy on the full stream
    model = CacheModel(tau=tau)
    results = {"zero-shot": (model.zero_shot(model.image_feats[stream])[1] == model.labels[stream]).float().cpu()}
    for name, eviction, slots in runs:
        model = CacheModel(eviction=eviction, slots=slots, tau=tau)
        correct = []
        for step, idx in enumerate(tqdm(stream, f"Running {name}", unit="images")):
            correct.append(int(model.predict([idx]).item() == model.labels[idx].item()))
            model.update(idx, step)
        results[name] = torch.tensor(correct, dtype=torch.float)

    print("\nonline accuracy (%) per segment")
    print(f"{'method':<12}{'all':>7}" + "".join(f"{s['domain'][:4] + s['visit'][-1]:>8}" for s in segments))
    for name, correct in results.items():
        row = "".join(f"{correct[s['start']:s['end']].mean().item() * 100:>8.2f}" for s in segments)
        print(f"{name:<12}{correct.mean().item() * 100:>7.2f}{row}")

if __name__ == "__main__":
    main(runs=[("entropy", "entropy", 3), ("cmr", "cmr", 3), ("entropy6", "entropy", 6), ("cmr6", "cmr", 6)],
         tau=0.3, check_steps=600)