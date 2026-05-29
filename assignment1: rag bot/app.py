import numpy as np
from pypdf import PdfReader
import google.generativeai as genai
import faiss 
from dotenv import load_dotenv 
import os

load_dotenv()

GEMINI_KEY=os.getenv('GEMINI_KEY')

genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")
pdf_path = "python_tutorial.pdf"

pdf_reader = PdfReader(pdf_path)

text = ""
for page in pdf_reader.pages:
    text += page.extract_text()

print("PDF loaded")

def chunk_text(text, size=800, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks

chunks = chunk_text(text)

print("Chunks:", len(chunks))

def do_embed(text):
    vec = np.zeros(384)

    for i, c in enumerate(text[:384]):
        vec[i % 384] += ord(c)

    return vec / (np.linalg.norm(vec) + 1e-9)

embeddings = np.array([do_embed(c) for c in chunks]).astype("float32")

index = faiss.IndexFlatL2(384)
index.add(embeddings)

print("FAISS Ready")

def retrieve(query, k=3):
    q_vec = do_embed(query).astype("float32").reshape(1, -1)

    _, idx = index.search(q_vec, k)

    return [chunks[i] for i in idx[0]]

def ask_gemini(question, context):
    prompt = f"""
You are an AI assistant.

Read and digest the qustions and Answer ONLY using the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    response = model.generate_content(prompt)
    return response.text

while True:
    q = input("\nAsk a question (type 'exit'): ")

    if q.lower() == "exit":
        break

    docs = retrieve(q)

    context = "\n".join(docs)

    answer = ask_gemini(q, context)

    print("\nAnswer:\n", answer)
