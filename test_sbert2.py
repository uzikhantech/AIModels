from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer, util
from mangum import Mangum
from typing import List

# Import sample data
import sample_data

# Initialize the FastAPI app
app = FastAPI()

# Load SBERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

@app.get("/similarity")
async def calculate_similarity():
    # Load sample data
    sentences = sample_data.sample_data

    # Check if there are enough sentences to compare
    if len(sentences) < 2:
        raise HTTPException(status_code=400, detail="At least two sentences are required.")

    # Generate embeddings
    embeddings = model.encode(sentences, convert_to_tensor=True)

    # Calculate cosine similarities
    cosine_scores = util.pytorch_cos_sim(embeddings, embeddings).cpu().numpy()

    # Convert cosine_scores to a list of lists for the response
    similarity_matrix = cosine_scores.tolist()

    # Display comparison results in a readable format
    comparison_results = []
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            comparison_results.append({
                "sentence1": sentences[i],
                "sentence2": sentences[j],
                "similarity_score": cosine_scores[i][j].item()
            })

    return {
        "similarity_matrix": similarity_matrix,
        "comparison_results": comparison_results
    }

# Integrate with AWS Lambda using Mangum
handler = Mangum(app)
