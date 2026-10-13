import torch
import torch.nn.functional as F
import open_clip
from pathlib import Path
from tqdm import tqdm
from data import OfficeHomeDataset
from config import clip_name, template
import math

class CacheModel:
    def __init__(self) -> None:
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

def main(taus: list) -> None:
    ds = OfficeHomeDataset(seed=0)
    model = CacheModel()
    model.extract_features(ds)

    _, preds, entropy, _ = model.zero_shot(model.image_feats)
    correct = (preds == model.labels).float()
    print(f"zero-shot accuracy, all images: {correct.mean().item() * 100:.2f}%")

    for domain in ds.domains:
        mask = torch.tensor([d == domain for d in model.domains], device=model.device)
        print(f"\n{domain}: zero-shot accuracy {correct[mask].mean().item() * 100:.2f}%, mean entropy {entropy[mask].mean().item():.3f}")
        for tau in taus:
            admitted = mask & (entropy < tau)
            # how many images of this domain would enter the cache, and how many of them have a correct label
            print(f"  tau {tau}: admitted {admitted.sum().item() / mask.sum().item() * 100:.1f}%, label accuracy of admitted {correct[admitted].mean().item() * 100:.1f}%")

if __name__ == "__main__":
    main([0.1, 0.2, 0.3, 0.4, 0.5])