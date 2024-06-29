import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import torch
import numpy as np
import pandas as pd
from mangum import Mangum

# Initialize the FastAPI app
app = FastAPI()

# Load the EfficientNet model
model = EfficientNet.from_pretrained('efficientnet-b0')
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_images(image_folder):
    images = []
    image_filenames = []
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            images.append(Image.open(image_path).convert("RGB"))
            image_filenames.append(filename)
    return images, image_filenames

def preprocess_image(image: Image.Image):
    return transform(image).unsqueeze(0)

@app.get("/similarity")
async def calculate_similarity():
    # Load images from the 'images' folder in the current directory
    image_folder = "./images"
    images, image_filenames = load_images(image_folder)

    if len(images) < 2:
        raise HTTPException(status_code=400, detail="At least two images are required in the 'images' folder.")

    # Generate embeddings
    image_embeddings = []
    with torch.no_grad():
        for image in images:
            input_tensor = preprocess_image(image)
            embedding = model.extract_features(input_tensor)
            embedding = embedding.mean([2, 3]).squeeze(0)  # Global Average Pooling
            image_embeddings.append(embedding)

    # Stack embeddings into a tensor
    image_embeddings = torch.stack(image_embeddings)

    # Normalize the embeddings
    image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)

    # Calculate cosine similarities
    cosine_similarity_matrix = (image_embeddings @ image_embeddings.T).cpu().numpy()

    # Create a DataFrame for readability
    df = pd.DataFrame(cosine_similarity_matrix, index=image_filenames, columns=image_filenames)

    # Format results
    similarity_results = {
        "similarity_matrix": df.to_dict()
    }

    return JSONResponse(content=similarity_results)

# Integrate with AWS Lambda using Mangum
handler = Mangum(app)
