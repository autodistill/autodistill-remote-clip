import os
from dataclasses import dataclass

import torch

import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel
from huggingface_hub import hf_hub_download
import open_clip
import numpy as np

from PIL import Image

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(os.path.join(HOME, ".autodistill/cache/remoteclip"), exist_ok=True)

checkpoint_path = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-ViT-B-32.pt", cache_dir=os.path.join(HOME, ".autodistill/cache/remoteclip"))

@dataclass
class RemoteCLIP(DetectionBaseModel):
    ontology: CaptionOntology
    
    def __init__(self, ontology: CaptionOntology):
        self.ontology = ontology

        self.model_name = "ViT-B-32"

        model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name)
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

        ckpt = torch.load(f"{checkpoint_path}", map_location=DEVICE)

        message = model.load_state_dict(ckpt)

        self.model = model.eval().to(DEVICE)

    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        prompts = self.ontology.prompts()

        text = self.tokenizer(prompts)
        image = self.preprocess(Image.open(input)).unsqueeze(0)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]

        return sv.Classifications(
            class_id=np.array([i for i in range(len(prompts))]),
            confidence=text_probs,
        )