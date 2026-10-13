import torch
import torch.nn.functional as F
import open_clip
from pathlib import Path
from tqdm import tqdm
from data import OfficeHomeDataset
from config import clip_name, template

class CacheModel:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cwd = Path.cwd()
        self.weights_path = Path(self.cwd, "models/vit_base_patch16_clip_224.openai/open_clip_pytorch_model.bin")
        self.feature_path = Path(self.cwd, "data/features_vitb16.pt")

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

def main() -> None:
    ds = OfficeHomeDataset(seed=0)
    model = CacheModel()
    model.extract_features(ds)

    feats = torch.load(model.feature_path)
    print(feats["image_feats"].shape, feats["text_feats"].shape, feats["labels"].shape)
    print(feats["image_feats"].norm(dim=-1)[:5])
    print(feats["classnames"][:5], feats["domains"][0], feats["domains"][-1])

if __name__ == "__main__":
    main()