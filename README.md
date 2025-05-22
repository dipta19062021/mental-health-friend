
Here is an updated version of the `README.md` that includes:

* ✅ A clear **use case**
* 📚 A breakdown and **exploration of the libraries used**
* 🗂️ Mention of text/PDF files and their purpose (knowledge base ingestion)

---

# 🧠 Mental Health Chatbot with Voice Assistant

**Buddy** is a compassionate AI-powered chatbot designed to support mental wellness. This application combines natural language processing, voice recognition, and document-based knowledge retrieval to simulate intelligent and empathetic conversations about mental health.

---

## 💡 Use Case

The chatbot is ideal for:

* Individuals seeking **mental health support** or emotional comfort
* Students learning how to combine **voice tech and LLMs** in a project
* Professionals exploring **document-based retrieval QA** systems
* Applications where **offline mental health guidance** is useful (e.g., rural, underserved areas)

Buddy reads and understands content from **text files or PDFs** related to mental well-being (e.g., therapy techniques, stress management guides, self-care documents), stores them in a **vector database**, and uses **retrieval-based QA** to answer user queries using this custom knowledge base.

---

## 📁 Data Files

Place your **PDF files** containing mental health information in the `/content/data/` folder.

Example files:

* `mental_health_basics.pdf`
* `anxiety_relief_tips.pdf`
* `cognitive_therapy_guide.pdf`

These files are processed and stored in a vector database for intelligent, context-aware answers.

---

## 🧰 Library Exploration

### 🔹 [Gradio](https://gradio.app/)

* Builds the web-based UI.
* Supports text, file, and **microphone input**.
* Easy to use for prototyping conversational AI apps.

### 🔹 [LangChain](https://www.langchain.com/)

* Core framework for building applications with language models.
* Used for:

  * `RetrievalQA` chain: for answering questions using vector DB
  * Custom prompt templates and context chaining

### 🔹 [Groq + LLaMA 3](https://groq.com/)

* High-performance large language model backend.
* `ChatGroq`: calls Groq's API for language generation.
* **LLaMA 3 70B**: Capable of nuanced, long-form conversational understanding.

### 🔹 [Chroma](https://www.trychroma.com/)

* Vector database used to store and retrieve document chunks.
* Provides fast semantic search over embedded PDF content.

### 🔹 [HuggingFace BGE Embeddings](https://huggingface.co/BAAI/bge-large-en-v1.5)

* Embeds text into dense vectors for similarity search.
* `bge-large-en-v1.5` is ideal for retrieval QA.

### 🔹 [gTTS](https://pypi.org/project/gTTS/)

* Google Text-to-Speech API.
* Converts chatbot replies into speech for voice interaction.

### 🔹 [pydub](https://github.com/jiaaro/pydub)

* Processes and plays audio files.
* Used for audio playback of TTS responses.

### 🔹 [SpeechRecognition](https://pypi.org/project/SpeechRecognition/)

* Converts microphone input (audio) into text.
* Enables **voice queries** from the user.

---

## 🚀 Quick Start

1. Add your PDFs to `/content/data/`
2. Install requirements:
   `pip install -r requirements.txt`
3. Launch the app:
   `python app.py`
4. Chat using text, voice, or document input!

---

## 🔐 Configuration

* Replace the placeholder in `groq_api_key='your_groq_api_key_here'` with your actual Groq API key.
* You may use `.env` and `python-dotenv` for better secret management.

---

## 📃 License

This project is licensed under the MIT License.

---

Let me know if you'd like me to split this into individual `.md` files or generate documentation per module.
