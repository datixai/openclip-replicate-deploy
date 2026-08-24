"""
OpenCLIP ViT-B/32 (LAION-2B) — Replicate model, packaged with Cog.

Loads the exact same checkpoint used to build the local gallery index
(laion/CLIP-ViT-B-32-laion2B-s34B-b79K), so embeddings returned here are
directly comparable to the ones already computed on Kaggle.

Returns a single L2-normalized embedding vector per image — ready to
compare against the gallery index with a plain dot product (cosine
similarity), no extra normalization needed on the caller's side.
"""

from typing import List

import torch
from PIL import Image
from cog import BasePredictor, Input, Path


HF_REPO = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Runs once when the container starts — loads the model into memory."""
        from transformers import CLIPModel, CLIPProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(HF_REPO).to(self.device).eval()
        # use_fast=False: falls back to the older, more stable image processor.
        # The newer "fast" processor (now the default in recent transformers
        # versions) can produce an inconsistent internal batch shape for a
        # single image, causing a confusing padding/tensor error.
        self.processor = CLIPProcessor.from_pretrained(HF_REPO, use_fast=False)

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
        # Pass a single image directly (not wrapped in a list) — this avoids
        # a batch-shape inconsistency bug seen with some transformers/CLIP
        # processor version combinations for single-image inputs.
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            raw_output = self.model.get_image_features(**inputs)

        feats = self.get_pooled_features(raw_output)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)  # matches how the gallery index was built

        return feats[0].cpu().tolist()