import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ------------------------------
# Load and preprocess documents
# ------------------------------
@st.cache_resource
def load_documents_and_index():
    with open("doctor_agent.txt", "r", encoding="utf-8") as f:
        medical_docs = [line.strip() for line in f.readlines() if line.strip()]

    # Embed documents
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = embedder.encode(medical_docs)
    dimension = doc_embeddings[0].shape[0]

    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(doc_embeddings))

    return medical_docs, embedder, index

# ------------------------------
# Retrieve top-k relevant docs
# ------------------------------
def retrieve_docs(query, embedder, index, docs, k=2):
    query_vector = embedder.encode([query])
    distances, indices = index.search(np.array(query_vector), k)
    return [docs[i] for i in indices[0]]

# ------------------------------
# Generate answer using Flan-T5
# ------------------------------
@st.cache_resource
def load_generator():
    return pipeline("text2text-generation", model="google/flan-t5-base")

def doctor_agent(query, embedder, index, docs, generator):
    context = " ".join(retrieve_docs(query, embedder, index, docs))
    prompt = f"Context: {context} \nQuestion: {query}"
    response = generator(prompt, max_length=100)[0]["generated_text"]
    return response

# ------------------------------
# Streamlit UI
# ------------------------------
def main():
    st.title("🩺 CareMate AI - Doctor Agent")
    st.markdown("Get intelligent, context-aware responses based on medical knowledge.")

    user_query = st.text_area("Describe your symptoms or medical question:")

    if st.button("Ask Doctor Agent"):
        if user_query.strip() == "":
            st.warning("Please enter a question or symptoms.")
        else:
            with st.spinner("Generating response..."):
                docs, embedder, index = load_documents_and_index()
                generator = load_generator()
                answer = doctor_agent(user_query, embedder, index, docs, generator)
                st.success("Doctor Agent Response:")
                st.write(answer)

if __name__ == "__main__":
    main()
