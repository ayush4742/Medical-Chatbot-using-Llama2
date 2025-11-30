from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import time
from collections import OrderedDict
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
index_name = os.environ.get('PINECONE_INDEX_NAME', 'medical-bot')

# --- Safety checks (optional but helpful) ---
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in .env")
if not index_name:
    raise ValueError("PINECONE_INDEX_NAME (Pinecone index name) is not set in .env")

# 1) Embeddings model  (all-MiniLM-L6-v2 → 384-dim)
embeddings = download_hugging_face_embeddings()

# 2) Initialize Pinecone client and index
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

# 3) Vector store wrapper
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

# simple in-memory LRU cache to speed up repeated queries
class LRUCache:
    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: str):
        v = self.cache.get(key)
        if v is not None:
            self.cache.move_to_end(key)
        return v

    def set(self, key: str, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


CACHE = LRUCache(capacity=300)

# 4) Prompt (imported from src.prompt)
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

# 5) Local Llama 2 model (keep temperature low to reduce hallucination)
llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={
        'max_new_tokens': 128,
        'temperature': 0.5,
        # optionally tighten sampling for more focused output
        'top_p': 0.9,
    },
)

# 6) Retrieval QA chain
 # lower retrieval size (k) to speed up the retrieval step (tradeoff: less context)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
)

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_msg = request.form["msg"]
    normalized = user_msg.strip()

    # return cached answer when available — very fast for repeated user queries
    cached = CACHE.get(normalized)
    if cached is not None:
        print("[cache hit]", normalized)
        return cached

    print("\n========================")
    print("User question:", user_msg)

    start = time.time()
    try:
        result = qa({"query": user_msg})
    except Exception as e:
        print("Error while generating answer:", e)
        return "Sorry, something went wrong while generating the answer."

    elapsed = time.time() - start
    print(f"Response time: {elapsed:.2f}s")

    answer = result["result"]
    sources = result.get("source_documents", [])

    print("Answer:", answer)
    print("---- Sources used ----")
    for i, doc in enumerate(sources, start=1):
        src = doc.metadata.get("source", "")
        snippet = doc.page_content[:300].replace("\n", " ")
        print(f"[{i}] {src} :: {snippet}...")
    print("========================\n")

    # cache the final answer for quick repeat responses
    CACHE.set(normalized, str(answer))

    return str(answer)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)