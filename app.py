from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import time
import json
import numpy as np
import faiss
from together import Together
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

st = ""
cs = [
 "people","work","life","person","good","always","year","decision","risk","education",
 "course","school","really","kind","job","family","child","someone","much","situation",
 "future","parent","help","first","lot","moment","come","army","thankful","naturally",
 "interviewer","informant","maybe","time","doe","somehow","likely","ever","thought",
 "originally","specifically","want","say","get","everything","right","general","well",
 "yes","like","can","couldn","okay","told","thank","now","example","understand",
 "being","think","probably","nothing","believe","question","make","know","own",
 "for example","all","consider","most","therefore","happen","didn","don","let",
 "got","often","way","also","went","see","take","wanted","just","one","still",
 "mean","even","will","something","thing","be","ask","type","as far as","point",
 "sight","allegedly","int","inf","either","whole","further"
]

md = SentenceTransformer("all-MiniLM-L6-v2")
ix = faiss.IndexFlatL2(384)

def tk(x):
    x = x.lower()
    x = re.sub(r"[^\w\s]"," ",x)
    w = x.split()
    w = [i for i in w if i not in cs]
    return w

@app.route("/")
def hm():
    return jsonify({"message":"Flask backend is running."})

@app.route("/upload_text", methods=["POST"])
def ut():
    global st
    d = request.json
    if not d or "text" not in d:
        return jsonify({"error":"No text field found"}),400
    st = d["text"]
    return jsonify({"message":"Text stored"}),200

@app.route("/preprocess", methods=["GET"])
def pp():
    global st
    if not st.strip():
        return jsonify({"error":"No transcripts found. Please upload first."}),400
    sn = re.split(r"(?<=[.?!])\s+", st)
    fm = {}
    cc = {}
    for s in sn:
        t = tk(s)
        for i in t:
            fm[i] = fm.get(i,0)+1
            if i not in cc:
                cc[i] = {}
        for i in range(len(t)):
            for j in range(i+1,len(t)):
                a = t[i]
                b = t[j]
                cc[a][b] = cc[a].get(b,0)+1
                if b not in cc:
                    cc[b] = {}
                cc[b][a] = cc[b].get(a,0)+1
    fa_ = sorted([{"word":k,"count":v} for k,v in fm.items()],key=lambda x:x["count"],reverse=True)
    nd = [{"id":i["word"]} for i in fa_]
    ln = []
    for a,n in cc.items():
        for b,v in n.items():
            if a<b and v>=2:
                ln.append({"source":a,"target":b,"value":v})
    return jsonify({"frequency":fa_,"graph":{"nodes":nd,"links":ln}})

@app.route("/run_advanced_model", methods=["POST"])
def ram():
    global st
    if not st.strip():
        return jsonify({"error":"No transcripts found. Please upload first."}),400
    ix.reset()
    em = md.encode([st])
    em = np.array(em).astype("float32")
    ix.add(em)
    ds,ig = ix.search(em,1)
    tc = Together()
    cp = ""
    p = [
      {
       "role":"system",
       "content":"You are a helpful assistant. Analyze the text below. Extract topics, entities, sentiments, themes, directions, outlook in a structured JSON format, then write a short summary incorporating these. Text: "+st
      }
    ]
    r = tc.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",
        messages=p,
        max_tokens=None,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>","<|eom_id|>"],
        stream=True
    )
    for t in r:
        if hasattr(t,"choices"):
            cp += t.choices[0].delta.content
    try:
        an = json.loads(cp)
    except:
        an = {"raw_response":cp}
    e = round(time.time(),2)
    return jsonify({"elapsed":e,"analysis":an})

if __name__=="__main__":
    app.run(debug=True)
