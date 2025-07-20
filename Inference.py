import os
import torch
import faiss
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, LlavaForConditionalGeneration
from pymongo import MongoClient

# --- CONFIG ---
MODEL_PATH = "workspace/llava_finetuned_2"
MONGO_URI = ""
DB_NAME = "productDb"
COLLECTION_NAME = "products_direct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM = 768  

# --- INIT ---
app = Flask(__name__)
mongo = MongoClient(MONGO_URI)[DB_NAME][COLLECTION_NAME]
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = LlavaForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32).to(DEVICE)
model.eval()

# --- FAISS INDEX ---
index = faiss.IndexFlatL2(EMBED_DIM)
id_map = {}  # maps FAISS index to MongoDB _id

# --- Helper: extract embedding from image + text ---
def get_embedding(image: Image.Image, text: str):
    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.vision_tower[0](inputs['pixel_values'])  # extract visual embedding
        vision_embed = outputs[:, 0, :]  # CLS token output

    return vision_embed.squeeze(0).cpu().numpy()  # shape (768,)

# --- Load existing product embeddings into FAISS ---
def load_embeddings():
    print("Loading embeddings into FAISS...")
    index.reset()
    id_map.clear()
    for i, doc in enumerate(mongo.find()):
        if "embedding" not in doc:
            continue
        emb = np.array(doc["embedding"]).astype("float32")
        index.add(np.expand_dims(emb, axis=0))
        id_map[i] = str(doc["_id"])
    print(f"Loaded {index.ntotal} vectors into FAISS")

# --- API: /match ---
@app.route("/match", methods=["POST"])
def match():
    if "image" not in request.files or "text" not in request.form:
        return jsonify({"error": "Missing image or text input"}), 400

    image = Image.open(BytesIO(request.files["image"].read())).convert("RGB")
    text = request.form["text"]

    query_emb = get_embedding(image, text).astype("float32").reshape(1, -1)
    D, I = index.search(query_emb, k=1)  # top 1 match

    if len(I[0]) == 0 or I[0][0] == -1:
        return jsonify({"error": "No match found"}), 404

    match_id = id_map[I[0][0]]
    match_doc = mongo.find_one({"_id": mongo.codec_options.document_class(match_id)})

    return jsonify({
        "matched_product": {
            "id": match_id,
            "title": match_doc.get("title"),
            "category": match_doc.get("category"),
            "description": match_doc.get("description"),
        },
        "distance": float(D[0][0])
    })

# --- API: /reload (refresh index) ---
@app.route("/reload", methods=["POST"])
def reload_index():
    load_embeddings()
    return jsonify({"status": "FAISS index reloaded", "total_vectors": index.ntotal})

# --- Optional: Ingest DB Embeddings (e.g., batch process them first) ---
@app.route("/batch_embed", methods=["POST"])
def batch_embed():
    updated = 0
    for doc in mongo.find():
        if "embedding" in doc:
            continue  # already has embedding

        try:
            image_path = doc["image_path"]
            image = Image.open(image_path).convert("RGB")
            text = doc.get("title", "") + " " + doc.get("category", "")
            emb = get_embedding(image, text)
            mongo.update_one({"_id": doc["_id"]}, {"$set": {"embedding": emb.tolist()}})
            updated += 1
        except Exception as e:
            print("Failed:", doc.get("_id"), str(e))
    return jsonify({"updated": updated})

# --- Start ---
if __name__ == "__main__":
    load_embeddings()
    app.run(host="0.0.0.0", port=8000)
