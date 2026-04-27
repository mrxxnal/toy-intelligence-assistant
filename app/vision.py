from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import io

# Load model + processor once (important for performance)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def convert_raw_image(raw_image):
    """
    Converts HuggingFace dataset image format → PIL Image
    """
    try:
        if isinstance(raw_image, dict) and "bytes" in raw_image:
            return Image.open(io.BytesIO(raw_image["bytes"])).convert("RGB")

        if isinstance(raw_image, Image.Image):
            return raw_image.convert("RGB")

        return raw_image
    except Exception:
        return None


def get_image_embedding(image):
    """
    Generates normalized CLIP image embedding (vision-only path)
    """
    image = convert_raw_image(image)

    if image is None:
        raise ValueError("Invalid image input for embedding generation")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        # 🔥 Use ONLY vision tower (avoid text encoder entirely)
        vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])

        # pooled representation of image
        embedding = vision_outputs.pooler_output

    # Normalize embedding (L2 normalization)
    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)

    return embedding.squeeze().cpu().numpy()