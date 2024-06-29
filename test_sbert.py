from fastapi import FastAPI, HTTPException, Request
from sentence_transformers import SentenceTransformer, util
from mangum import Mangum
import json

# Initialize the FastAPI app
app = FastAPI()

# Load SBERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

@app.post("/similarity")
async def calculate_similarity(request: Request):
    # Extract JSON body from request
    try:
        body = await request.json()
        sentences = body["sentences"]
    except (json.JSONDecodeError, KeyError):
        raise HTTPException(status_code=400, detail="Invalid input. Ensure the body is a JSON object with a 'sentences' key.")

    # Check if there are enough sentences to compare
    if len(sentences) < 2:
        raise HTTPException(status_code=400, detail="At least two sentences are required.")

    # Generate embeddings
    embeddings = model.encode(sentences, convert_to_tensor=True)

    # Calculate cosine similarities
    cosine_scores = util.pytorch_cos_sim(embeddings, embeddings).cpu().numpy()

    # Convert cosine_scores to a list of lists for the response
    similarity_matrix = cosine_scores.tolist()

    return {"similarity_matrix": similarity_matrix}

# Integrate with AWS Lambda using Mangum
handler = Mangum(app)
