from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import re
import time
import json
import numpy as np
import faiss
import nltk
from nltk.corpus import stopwords
from together import Together
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
nltk.download('stopwords') 
nltk_stop = set(stopwords.words('english'))  

st = "" 
cs = {
    "people","work","life","person","good","always","year","decision","risk","education","course","school","really",
    "kind","job","family","child","someone","much","situation","future","parent","help","first","lot","moment","come",
    "army","thankful","naturally","interviewer","informant","maybe","time","doe","somehow","likely","ever","thought",
    "originally","specifically","want","say","get","everything","right","general","well","yes","like","can","couldn",
    "okay","told","thank","now","example","understand","being","think","probably","nothing","believe","question","make",
    "know","own","for example","all","consider","most","therefore","happen","didn","don","let","got","often","way",
    "also","went","see","take","wanted","just","one","still","mean","even","will","something","thing","be","ask","type",
    "as far as","point","sight","allegedly","int","inf","either","whole","further"
}
all_stops = nltk_stop.union(cs)

md = SentenceTransformer("all-MiniLM-L6-v2")
ix = faiss.IndexFlatL2(384)

def tk(x):
    x = x.lower()
    x = re.sub(r"[^\w\s]", " ", x)
    w = x.split()
    w = [i for i in w if i not in all_stops]
    return w
@app.route("/")
def home():
    return jsonify({"message": "Flask backend is running."})

@app.route("/upload_text", methods=["POST"])
def upload_text():
    global st
    d = request.json
    if not d or "text" not in d:
        return jsonify({"error": "No text field found"}), 400
    st = d["text"]
    return jsonify({"message": "Text stored"}), 200

@app.route("/preprocess", methods=["GET"])
def preprocess():
    global st
    if not st.strip():
        return jsonify({"error": "No transcripts found. Please upload first."}), 400

    sentences = re.split(r"(?<=[.?!])\s+", st)
    freq_map = {}
    co_occur = {}
    for s_ in sentences:
        tokens = tk(s_)
        for tok in tokens:
            freq_map[tok] = freq_map.get(tok, 0) + 1
            if tok not in co_occur:
                co_occur[tok] = {}
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                a, b = tokens[i], tokens[j]
                co_occur[a][b] = co_occur[a].get(b, 0) + 1
                if b not in co_occur:
                    co_occur[b] = {}
                co_occur[b][a] = co_occur[b].get(a, 0) + 1

    freq_arr = sorted(
        [{"word": k, "count": v} for k, v in freq_map.items()],
        key=lambda x: x["count"],
        reverse=True
    )
    nodes = [{"id": item["word"]} for item in freq_arr]
    links = []
    for a, nbrs in co_occur.items():
        for b, val in nbrs.items():
            if a < b and val >= 2:
                links.append({"source": a, "target": b, "value": val})

    return jsonify({
        "frequency": freq_arr,
        "graph": {"nodes": nodes, "links": links}
    })

@app.route("/run_advanced_model", methods=["POST"])
def run_advanced_model():
    global st
    if not st.strip():
        return jsonify({"error": "No transcripts found. Please upload first."}), 400

    ix.reset()
    emb = md.encode([st])
    emb = np.array(emb).astype("float32")
    ix.add(emb)
    tc = Together(api_key="6a98a50ecc91a4b4ced67b24f1376e1ba96a35b18a017d9f45b42049be0ad611")

    p = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Analyze the text below. "
                "Extract topics, entities, sentiments, themes, directions, outlook in a structured JSON format, "
                "then write a short summary incorporating these. Also generate any needed Python code if relevant, "
                "under a key named 'generated_code'. The text is:\n\n" + st
            )
        }
    ]

    t0 = time.time()
    response_text = ""
    r_ = tc.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",
        messages=p,
        max_tokens=None,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>", "<|eom_id|>"],
        stream=True
    )
    for chunk in r_:
        if hasattr(chunk, "choices"):
            response_text += chunk.choices[0].delta.content

    elapsed = round(time.time() - t0, 2)
    try:
        analysis = json.loads(response_text)
    except:
        analysis = {"raw_response": response_text}
    with open("analysis_output.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    if isinstance(analysis, dict) and "generated_code" in analysis:
        with open("generated_code.py", "w", encoding="utf-8") as cf:
            cf.write(analysis["generated_code"])

    return jsonify({"elapsed": elapsed, "analysis": analysis})

@app.route("/download_analysis", methods=["GET"])
def download_analysis():
    return send_file("analysis_output.json", as_attachment=True)

@app.route("/download_code", methods=["GET"])
def download_code():
    return send_file("generated_code.py", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
