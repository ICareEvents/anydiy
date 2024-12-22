from flask import Flask, request, jsonify
import re
import random
import time
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)

# Set up CORS to allow requests from the frontend
CORS(app, resources={r"/*": {"origins": ["https://crisil-one.vercel.app"]}})

# Global storage for transcripts
STORE_TEXT = ""

# Custom stopword list
CUSTOM_STOPWORDS = [
    "people", "work", "life", "person", "good", "always", "year", "decision", "risk", "education", "course",
    "school", "really", "kind", "job", "family", "child", "someone", "much", "situation", "future", "parent",
    "help", "first", "lot", "moment", "come", "army", "thankful", "naturally", "interviewer", "informant"
]

# Utility function to tokenize and remove stopwords
def tokenize_and_remove_stopwords(text):
    low = text.lower()
    low = re.sub(r"[^\w\s]", " ", low)  # Remove punctuation
    words = low.split()
    filtered = [w for w in words if w not in CUSTOM_STOPWORDS]
    return filtered

@app.route("/")
def home():
    return jsonify({"message": "Flask backend is running."})

@app.route("/upload_text", methods=["OPTIONS", "POST"])
def upload_text():
    if request.method == "OPTIONS":
        # Handle preflight request
        response = jsonify({"message": "CORS preflight passed"})
        response.headers.add("Access-Control-Allow-Origin", "https://crisil-one.vercel.app")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        return response, 200

    # Handle POST request
    global STORE_TEXT
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "No text field found"}), 400
    STORE_TEXT = data["text"]
    response = jsonify({"message": "Text stored"})
    response.headers.add("Access-Control-Allow-Origin", "https://crisil-one.vercel.app")
    return response, 200

@app.route("/preprocess", methods=["GET"])
def preprocess():
    try:
        global STORE_TEXT
        if not STORE_TEXT.strip():
            return jsonify({"error": "No transcripts found. Please upload first."}), 400

        # Split text into sentences and process
        sentences = re.split(r"(?<=[.?!])\s+", STORE_TEXT)
        freq_map = {}
        cooccur = {}

        for sen in sentences:
            tokens = tokenize_and_remove_stopwords(sen)
            for t in tokens:
                freq_map[t] = freq_map.get(t, 0) + 1
                if t not in cooccur:
                    cooccur[t] = {}
            for i in range(len(tokens)):
                for j in range(i + 1, len(tokens)):
                    wA, wB = tokens[i], tokens[j]
                    cooccur[wA][wB] = cooccur[wA].get(wB, 0) + 1
                    cooccur[wB] = cooccur.get(wB, {})
                    cooccur[wB][wA] = cooccur[wB].get(wA, 0) + 1

        # Prepare frequency data
        freq_arr = sorted(
            [{"word": k, "count": v} for k, v in freq_map.items()],
            key=lambda x: x["count"],
            reverse=True
        )

        # Build graph data
        nodes = [{"id": item["word"]} for item in freq_arr]
        links = []
        for wA, neighbors in cooccur.items():
            for wB, val in neighbors.items():
                if wA < wB and val >= 2:
                    links.append({"source": wA, "target": wB, "value": val})

        return jsonify({
            "frequency": freq_arr,
            "graph": {
                "nodes": nodes,
                "links": links
            }
        }), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/run_advanced_model", methods=["POST"])
def run_advanced_model():
    try:
        global STORE_TEXT
        if not STORE_TEXT.strip():
            return jsonify({"error": "No transcripts found. Please upload first."}), 400

        # Start timer
        start = time.time()

        # Mock advanced model outputs
        results = [
            {
                "model": "LDA",
                "coherence": round(random.uniform(0.38, 0.42), 3),
                "time_sec": random.randint(15, 25),
                "topic_diversity": random.randint(15, 30),
                "umass": round(random.uniform(-2, -1), 3),
                "npmi": round(random.uniform(-0.3, -0.1), 3),
                "uci": round(random.uniform(8, 9), 3),
                "silhouette": round(random.uniform(0.2, 0.35), 3),
                "dbcv": None
            },
            {
                "model": "BERT+Kmeans",
                "coherence": round(random.uniform(0.40, 0.45), 3),
                "time_sec": random.randint(35, 60),
                "topic_diversity": random.randint(25, 40),
                "umass": round(random.uniform(-3.2, -2.5), 3),
                "npmi": round(random.uniform(-0.2, -0.1), 3),
                "uci": round(random.uniform(8, 10), 3),
                "silhouette": round(random.uniform(0.4, 0.55), 3),
                "dbcv": None
            },
        ]

        # Select the best model
        best_model = max(results, key=lambda x: x["coherence"])

        # Mock topics
        mock_topics = {
            "Topic 0": ["work", "study", "career", "electrician"],
            "Topic 1": ["good", "knowledge", "study", "saying"],
        }

        elapsed = round(time.time() - start, 2)
        return jsonify({
            "elapsed": elapsed,
            "results": {
                "models": results,
                "best_model": best_model,
                "topics": mock_topics
            }
        }), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Use Render's PORT variable or default to 5000
    app.run(host="0.0.0.0", port=port)
