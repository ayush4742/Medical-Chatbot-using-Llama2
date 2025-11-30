from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME')

# --- Safety checks (optional but helpful) ---
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in .env")
if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME is not set in .env")

# 1) Embeddings model  (all-MiniLM-L6-v2 → 384-dim)
embeddings = download_hugging_face_embeddings()

# 2) Initialize Pinecone client and index
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# 3) Vector store wrapper
docsearch = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)

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
        "max_new_tokens": 200,       # chhota output
        "temperature": 0.1,          # zyada random nahi
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.2    # repeat kam karega
    }
)

# 6) Retrieval QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 5}),  # was k=2
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
)

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_msg = request.form["msg"]

    print("\n========================")
    print("User question:", user_msg)

    try:
        result = qa({"query": user_msg})
    except Exception as e:
        print("Error while generating answer:", e)
        return "Sorry, something went wrong while generating the answer."

    answer = result["result"]
    sources = result.get("source_documents", [])

    print("Answer:", answer)
    print("---- Sources used ----")
    for i, doc in enumerate(sources, start=1):
        src = doc.metadata.get("source", "")
        snippet = doc.page_content[:300].replace("\n", " ")
        print(f"[{i}] {src} :: {snippet}...")
    print("========================\n")

    return str(answer)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)