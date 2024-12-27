from flask import Flask, request, jsonify
import re
import random
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from all origins

# We'll store the transcripts in memory for this demo
STORE_TEXT = ""

# A custom stopword list (per your request)
CUSTOM_STOPWORDS = [
    "people","work","life","person","good","always","year","decision","risk","education","course",
    "school","really","kind","job","family","child","someone","much","situation","future","parent",
    "help","first","lot","moment","come","army","thankful","naturally","interviewer","informant"
]

def tokenize_and_remove_stopwords(text):
    """Lowercase, remove punctuation, split, remove custom stopwords."""
    low = text.lower()
    low = re.sub(r"[^\w\s]", " ", low)  # remove punctuation
    words = low.split()
    filtered = [w for w in words if w not in CUSTOM_STOPWORDS]
    return filtered

@app.route("/")
def home():
    """Sanity check: Just returns a message."""
    return jsonify({"message": "Flask backend is running."})

@app.route("/upload_text", methods=["POST"])
def upload_text():
    """Uploads transcript text from the frontend, stores in memory."""
    global STORE_TEXT
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "No text field found"}), 400
    STORE_TEXT = data["text"]
    return jsonify({"message": "Text stored"}), 200

@app.route("/preprocess", methods=["GET"])
def preprocess():
    """Preprocesses the stored text, returns word frequencies + co-occurrence graph."""
    global STORE_TEXT
    if not STORE_TEXT.strip():
        return jsonify({"error": "No transcripts found. Please upload first."}), 400

    # Naive split into sentences
    sentences = re.split(r"(?<=[.?!])\s+", STORE_TEXT)
    freq_map = {}
    cooccur = {}

    # Build frequency map + co-occurrence
    for sen in sentences:
        tokens = tokenize_and_remove_stopwords(sen)
        for t in tokens:
            freq_map[t] = freq_map.get(t, 0) + 1
            if t not in cooccur:
                cooccur[t] = {}
        # co-occurrence
        for i in range(len(tokens)):
            for j in range(i+1, len(tokens)):
                wA = tokens[i]
                wB = tokens[j]
                cooccur[wA][wB] = cooccur[wA].get(wB, 0) + 1
                if wB not in cooccur:
                    cooccur[wB] = {}
                cooccur[wB][wA] = cooccur[wB].get(wA, 0) + 1

    # Sort frequencies descending
    freq_arr = sorted(
        [{"word": k, "count": v} for k, v in freq_map.items()],
        key=lambda x: x["count"],
        reverse=True
    )

    # Build node-link structure
    nodes = [{"id": item["word"]} for item in freq_arr]
    links = []
    for wA, neighbors in cooccur.items():
        for wB, val in neighbors.items():
            # only add link if wA < wB (avoid duplication) and frequency >= 2
            if wA < wB and val >= 2:
                links.append({
                    "source": wA,
                    "target": wB,
                    "value": val
                })

    return jsonify({
        "frequency": freq_arr,
        "graph": {
            "nodes": nodes,
            "links": links
        }
    })

@app.route("/run_advanced_model", methods=["POST"])
def run_advanced_model():
    """Mocks out a long-running advanced modeling pipeline and returns random results."""
    global STORE_TEXT
    if not STORE_TEXT.strip():
        return jsonify({"error": "No transcripts found. Please upload first."}), 400

    start = time.time()

    # Fake results for demonstration
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
        {
            "model": "BERT+LDA+KMeans",
            "coherence": round(random.uniform(0.43, 0.47), 3),
            "time_sec": random.randint(50, 80),
            "topic_diversity": random.randint(30, 50),
            "umass": round(random.uniform(-3.5, -3), 3),
            "npmi": round(random.uniform(-0.3, -0.1), 3),
            "uci": round(random.uniform(8, 10), 3),
            "silhouette": round(random.uniform(0.4, 0.6), 3),
            "dbcv": None
        },
        {
            "model": "BERT+LDA+HDBSCAN",
            "coherence": round(random.uniform(0.45, 0.48), 3),
            "time_sec": random.randint(60, 90),
            "topic_diversity": random.randint(35, 55),
            "umass": round(random.uniform(-3.6, -3.1), 3),
            "npmi": round(random.uniform(-0.3, -0.1), 3),
            "uci": round(random.uniform(8, 10), 3),
            "silhouette": None,
            "dbcv": round(random.uniform(0.5, 0.7), 3)
        },
        {
            "model": "BERT+HDBSCAN",
            "coherence": round(random.uniform(0.48, 0.50), 3),
            "time_sec": random.randint(60, 90),
            "topic_diversity": random.randint(40, 60),
            "umass": None,
            "npmi": None,
            "uci": None,
            "silhouette": None,
            "dbcv": round(random.uniform(0.5, 0.8), 3)
        },
    ]

    best_model = max(results, key=lambda x: x["coherence"])
    mock_topics = {
        "Topic 0": ["work", "to", "study", "career", "electrician"],
        "Topic 1": ["good", "knowledge", "to", "study", "saying"],
        "Topic 2": ["risk", "child", "education", "do", "receive"],
        "Topic 3": ["go", "job", "favorite", "business", "nobody"]
    }

    elapsed = round(time.time() - start, 2)
    return jsonify({
        "elapsed": elapsed,
        "results": {
            "models": results,
            "best_model": best_model,
            "topics": mock_topics
        }
    })

if __name__ == "__main__":
    # To run locally at http://127.0.0.1:5000/
    app.run(debug=True)
