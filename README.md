# 🩺 CareMate AI — Doctor Agent

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0078D4?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Flan--T5-FFD21F?style=flat-square&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> An intelligent medical assistant powered by **RAG (Retrieval-Augmented Generation)** — combines FAISS vector search with Flan-T5 to deliver context-aware responses to health queries.

---

## 🧠 About the Project

CareMate AI is a lightweight **RAG-based doctor agent** that answers medical questions by first retrieving the most relevant context from a medical knowledge base, then generating a natural language response using Google's **Flan-T5** language model.

Unlike a plain chatbot, CareMate grounds its answers in retrieved medical documents — reducing hallucinations and improving response relevance.

---

## ✨ Features

- 🔍 **Semantic Search** — FAISS index finds the most relevant medical documents for any query
- 🤖 **LLM Response Generation** — Flan-T5 generates context-aware, human-readable answers
- 🧬 **RAG Pipeline** — Retrieval-Augmented Generation for grounded, accurate responses
- ⚡ **Cached Models** — `@st.cache_resource` ensures fast repeated queries
- 🖥️ **Clean Streamlit UI** — Simple symptom input and instant response display

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.8+ |
| Vector Search | FAISS (IndexFlatL2) |
| Embeddings | SentenceTransformer (`all-MiniLM-L6-v2`) |
| LLM | Google Flan-T5 Base (HuggingFace) |
| Frontend | Streamlit |
| Numerics | NumPy |

---

## 📁 Project Structure

```
Doctor_Agent/
│
├── app.py               # Main Streamlit app — RAG pipeline & UI
├── doctor_agent.txt     # Medical knowledge base (document store)
└── requirements.txt     # Project dependencies
```

---

## ⚙️ How It Works — RAG Pipeline

```
User Query (symptoms / question)
          ↓
  SentenceTransformer Embedding
          ↓
  FAISS Index Search (Top-K docs)
          ↓
  Retrieved Context + Query → Prompt
          ↓
  Flan-T5 Text Generation
          ↓
  Context-Aware Medical Response
```

1. Medical documents from `doctor_agent.txt` are embedded using `all-MiniLM-L6-v2`
2. Embeddings are stored in a **FAISS IndexFlatL2** for fast similarity search
3. On a user query, the top-2 most relevant documents are retrieved
4. A prompt is constructed: `Context: <retrieved docs> \nQuestion: <user query>`
5. **Flan-T5** generates a natural language response from the prompt

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Doctor_Agent.git
cd Doctor_Agent

# 2. Install dependencies
pip install streamlit numpy faiss-cpu sentence-transformers transformers

# 3. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🖥️ Usage

1. Launch the app with `streamlit run app.py`
2. Type your symptoms or medical question in the text area
   - Example: *"I have a persistent cough and feel tired all the time"*
3. Click **"Ask Doctor Agent"**
4. CareMate AI retrieves relevant medical context and generates a response

---

## 💡 Example Interactions

| Input | Response |
|-------|----------|
| *"I have a sore throat and fever"* | Suggests possible viral infection based on retrieved context |
| *"Stomach pain and vomiting after eating"* | Points toward food poisoning symptoms |
| *"Chest pain after working out"* | Suggests possible muscle strain |

---

## 🔧 Extending the Knowledge Base

To add more medical knowledge, simply append new lines to `doctor_agent.txt`:

```
Headache with sensitivity to light may indicate a migraine.
High blood pressure with dizziness could be hypertension-related.
```

No retraining needed — the FAISS index rebuilds on next app launch.

---

## ⚠️ Disclaimer

> CareMate AI is built for **educational purposes only**. It is **not a licensed medical tool** and must **not replace professional medical advice**. Always consult a qualified doctor for health concerns.

---

## 📦 Dependencies

```
streamlit
numpy
faiss-cpu
sentence-transformers
transformers
```

---

## 🙋‍♂️ Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

⭐ **If you found this project helpful, please give it a star!**
