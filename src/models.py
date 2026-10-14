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
    def __init__(self, alpha: float = 2.0, beta: float = 5.0) -> None:
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

def main(alpha: float, beta: float, shots: int) -> None:
    ds = OfficeHomeDataset(seed=0)
    model = CacheModel(alpha=alpha, beta=beta)
    heldout = [i for d in ds.domains for i in ds.splits[d]["heldout"]]
    visit1 = [i for d in ds.domains for i in ds.splits[d]["visit1"]]

    zs_preds = model.zero_shot(model.image_feats[heldout])[1]
    print(f"\nalpha {alpha}, beta {beta}, {shots} shots per class")
    print(f"zero-shot heldout accuracy: {(zs_preds == model.labels[heldout]).float().mean().item() * 100:.2f}%")
    print(f"empty cache heldout accuracy: {(model.predict(heldout) == model.labels[heldout]).float().mean().item() * 100:.2f}%")

    # fill the cache by hand with a few visit1 images per class, once with true labels and once with zero-shot labels
    rng = random.Random(0)
    picked = []
    for c in range(model.num_classes):
        pool = [i for i in visit1 if model.labels[i].item() == c and model.zero_shot(model.image_feats[[i]])[2].item() < 0.3]
        picked += rng.sample(pool, shots)

    for label_type in ["true", "zero-shot"]:
        model.reset()
        model.cache["ids"] = picked
        if label_type == "true":
            model.cache["labels"] = model.labels[picked].tolist()
        else:
            model.cache["labels"] = model.zero_shot(model.image_feats[picked])[1].tolist()
        acc = (model.predict(heldout) == model.labels[heldout]).float().mean().item()
        print(f"cache with {label_type} labels, heldout accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    for alpha in [0.25, 0.5, 1.0, 2.0]:
        main(alpha=alpha, beta=5.0, shots=3)