"""
OpenCLIP ViT-B/32 (LAION-2B) — Replicate model, packaged with Cog.

Loads the exact same checkpoint used to build the local gallery index
(laion/CLIP-ViT-B-32-laion2B-s34B-b79K), so embeddings returned here are
directly comparable to the ones already computed on Kaggle.

Returns a single L2-normalized embedding vector per image — ready to
compare against the gallery index with a plain dot product (cosine
similarity), no extra normalization needed on the caller's side.

NOTE ON IMAGE PREPROCESSING: this deliberately does NOT use CLIPProcessor's
built-in image preprocessing. That path threw a reproducible tensor error
for this specific checkpoint/transformers-version combination (an internal
bug in how the processor's preprocess() builds its output tensor — not
something fixable from calling code). Instead, image preprocessing is done
manually here using CLIP's standard, well-documented recipe: resize to
224px, center-crop, normalize with CLIP's known mean/std constants. This
is the same math CLIPProcessor performs internally when it works correctly
— it just skips the specific code path that was failing.
"""

from typing import List

import torch
from PIL import Image
from torchvision import transforms
from cog import BasePredictor, Input, Path


HF_REPO = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

# Standard CLIP preprocessing constants — the same values used by OpenAI's
# original CLIP and by LAION's OpenCLIP checkpoints (they share the same
# preprocessing recipe).
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMG_SIZE = 224

preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Runs once when the container starts — loads the model into memory."""
        from transformers import CLIPModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(HF_REPO).to(self.device).eval()

    def get_pooled_features(self, output):
        """Handles whatever shape get_image_features() returns across
        different transformers versions — a plain tensor, or a wrapped
        output object. Same defensive helper used in the local test site."""
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "image_embeds") and output.image_embeds is not None:
            return output.image_embeds
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state.mean(dim=1)
        raise TypeError(f"Unrecognized model output type: {type(output)}")

    def predict(
        self,
        image: Path = Input(description="Photo of the item to embed"),
    ) -> List[float]:
        """Returns one L2-normalized embedding vector (512 floats) for the image."""
        pil_image = Image.open(image).convert("RGB")

        # Manual preprocessing — bypasses CLIPProcessor's built-in image
        # pipeline entirely (see module docstring for why).
        pixel_values = preprocess(pil_image).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)

        with torch.no_grad():
            raw_output = self.model.get_image_features(pixel_values=pixel_values)

        feats = self.get_pooled_features(raw_output)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)  # matches how the gallery index was built

        return feats[0].cpu().tolist()