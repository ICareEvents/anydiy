from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import re
import time
import json
import numpy as np
import faiss
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from together import Together
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

st = ""
nltk_stop = set(stopwords.words('english'))

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
    """
    Tokenizer to lowercase, remove punctuation, split, and remove both
    NLTK + custom stopwords.
    """
    x = x.lower()
    x = re.sub(r"[^\w\s]", " ", x)
    w = x.split()
    w = [i for i in w if i not in all_stops]
    return w

@app.route("/")
def hm():
    return jsonify({"message": "Flask backend is running with NLTK stopwords."})

@app.route("/upload_text", methods=["POST"])
def ut():
    """Store the uploaded text in a global variable."""
    global st
    d = request.json
    if not d or "text" not in d:
        return jsonify({"error": "No text field found"}), 400
    st = d["text"]
    return jsonify({"message": "Text stored"}), 200

@app.route("/preprocess", methods=["GET"])
def pp():
    """
    Preprocess the stored text, get word frequencies + co-occurrence graph.
    """
    global st
    if not st.strip():
        return jsonify({"error": "No transcripts found. Please upload first."}), 400

    sn = re.split(r"(?<=[.?!])\s+", st)
    fm = {}
    cc = {}
    for s_ in sn:
        t_ = tk(s_)
        for i_ in t_:
            fm[i_] = fm.get(i_, 0) + 1
            if i_ not in cc:
                cc[i_] = {}
        for i in range(len(t_)):
            for j in range(i+1, len(t_)):
                a = t_[i]
                b = t_[j]
                cc[a][b] = cc[a].get(b, 0) + 1
                if b not in cc:
                    cc[b] = {}
                cc[b][a] = cc[b].get(a, 0) + 1

    fa_ = sorted(
        [{"word": k, "count": v} for k, v in fm.items()],
        key=lambda x: x["count"],
        reverse=True
    )
    nd = [{"id": i["word"]} for i in fa_]
    ln = []
    for a, n in cc.items():
        for b, v in n.items():
            if a < b and v >= 2:
                ln.append({"source": a, "target": b, "value": v})

    return jsonify({
        "frequency": fa_,
        "graph": {"nodes": nd, "links": ln}
    })

@app.route("/run_advanced_model", methods=["POST"])
def ram():
    """
    1) Encode st
    2) LLM extracts topics, entities, sentiments, etc. in structured JSON
    3) Summarizes the text
    4) Potentially returns generated code in the JSON
    5) Save results in 'analysis_output.json'
    6) If 'generated_code' is present, also save 'generated_code.py'
    7) Return JSON with analysis + time
    """
    global st
    if not st.strip():
        return jsonify({"error": "No transcripts found. Please upload first."}), 400
    ix.reset()
    emb = md.encode([st])
    emb = np.array(emb).astype("float32")
    ix.add(emb)
    tc = Together(api_key="6a98a50ecc91a4b4ced67b24f1376e1ba96a35b18a017d9f45b42049be0ad611")

    cp = ""
    p = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Analyze the text below. "
                "Extract topics, entities, sentiments, themes, directions, outlook in a structured JSON format, "
                "then write a short summary incorporating these. Also generate any needed Python code if relevant, "
                "in a key named 'generated_code'. The text is: \n\n" + st
            )
        }
    ]

    start_time = time.time()

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
    for tok in r_:
        if hasattr(tok, "choices"):
            cp += tok.choices[0].delta.content

    elapsed_time = round(time.time() - start_time, 2)

    try:
        an = json.loads(cp)
    except:
        an = {"raw_response": cp}
    with open("analysis_output.json", "w", encoding="utf-8") as f:
        json.dump(an, f, indent=2, ensure_ascii=False)
    if isinstance(an, dict) and "generated_code" in an:
        code_content = an["generated_code"]
        with open("generated_code.py", "w", encoding="utf-8") as cf:
            cf.write(code_content)

    return jsonify({"elapsed": elapsed_time, "analysis": an})

@app.route("/download_analysis", methods=["GET"])
def download_analysis():
    """
    If you want to let the user download the 'analysis_output.json' directly.
    """
    return send_file("analysis_output.json", as_attachment=True)

@app.route("/download_code", methods=["GET"])
def download_code():
    """
    If you want to let the user download 'generated_code.py' if it exists.
    """
    return send_file("generated_code.py", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
