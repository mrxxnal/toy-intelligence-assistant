from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import io

# Load model + processor once
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def convert_raw_image(raw_image):
    """
    Converts HuggingFace dataset image format → PIL Image
    """
    if isinstance(raw_image, dict) and "bytes" in raw_image:
        return Image.open(io.BytesIO(raw_image["bytes"])).convert("RGB")
    
    return raw_image


def get_image_embedding(image):
    image = convert_raw_image(image)

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        # 🔥 Use ONLY vision model (bypass text completely)
        vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])

        # Take pooled output
        embedding = vision_outputs.pooler_output

    # Normalize
    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)

    return embedding.squeeze().cpu().numpy()