import gradio as gr
import time
import os
import io
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === LLM + VectorDB Setup ===
def initialize_llm():
    llm = ChatGroq(
        temperature=0,
        groq_api_key='YOUR API KEY',
        model="llama-3.3-70b-versatile"
    )
    return llm

def create_vector_db():
    loader = DirectoryLoader("/content/data", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory="./Chroma_db")
    vector_db.persist()
    print("Chroma db created and saved successfully")
    return vector_db

def setup_q_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_template = """You are a mental health chatbot. Respond through this:
    {context}
    User: {question}
    Chatbot: """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# === TTS Function ===
def speak_text(text):
    tts = gTTS(text)
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    audio = AudioSegment.from_file(audio_fp, format="mp3")
    play(audio)

# === Initialize System ===
print("Initializing Chatbot.........")
llm = initialize_llm()
db_path = "/content/Chroma_db"

if not os.path.exists(db_path):
    print("Creating Vector Database.........")
    vector_db = create_vector_db()
else:
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

qa_chain = setup_q_chain(vector_db, llm)

# === Chat logic with voice input ===
def add_message(history, message, audio):
    if audio:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                history.append({"role": "user", "content": text})
            except sr.UnknownValueError:
                history.append({"role": "user", "content": "Sorry, I couldn't understand that. Please try typing."})
    elif message["text"]:
        history.append({"role": "user", "content": message["text"]})
    
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def bot(history):
    user_msg = history[-1]["content"]
    doctor_emoji = "🧑‍⚕️"

    if isinstance(user_msg, dict):
        reply = f"{doctor_emoji} I received your file, but I can't process it yet. Please ask me something."
    else:
        greeting_triggers = ["hi", "hello", "hey", "hii", "heyy", "good morning", "good evening", "good afternoon"]
        lowered_msg = user_msg.lower().strip()

        # Show welcome message only for the first user message
        is_first_message = sum(1 for msg in history if msg["role"] == "user") == 1

        if any(greet in lowered_msg for greet in greeting_triggers) and is_first_message:
            reply = f"{doctor_emoji} Hi there! I am **Buddy**, your friendly mental wellness companion. How can I help you today?"
        else:
            bot_reply = qa_chain.run(user_msg)
            reply = f"{doctor_emoji} {bot_reply}"

    history.append({"role": "assistant", "content": ""})
    for char in reply:
        history[-1]["content"] += char
        time.sleep(0.02)
        yield history

    speak_text(reply)

# === Custom Calm UI Theme (CSS) ===
calm_css = """
.gradio-container {
    max-width: 800px;
    margin: auto;
    font-family: 'Segoe UI', sans-serif;
    background: #f0fdf4;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 0 12px rgba(0, 100, 80, 0.1);
}
#chatbot {
    border: 1px solid #cce3d6;
    border-radius: 16px;
    background-color: #ffffff;
    padding: 10px;
}
#chatbot .message.user {
    background-color: #e1f5ec;
    border-radius: 12px;
    padding: 8px 12px;
    color: #14532d;
}
#chatbot .message.assistant {
    background-color: #e0f2fe;
    border-radius: 12px;
    padding: 8px 12px;
    color: #0c4a6e;
}
h1 {
    color: #14532d;
    text-align: center;
    font-size: 2.2em;
    margin-bottom: 8px;
}
.gr-markdown {
    color: #334d4d;
    text-align: center;
    font-size: 1.1em;
    margin-bottom: 20px;
}
"""

# === Gradio UI with Microphone ===
with gr.Blocks(css=calm_css) as app:
    gr.Markdown("<h1>🧠 Mental Health Chatbot</h1>")
    gr.Markdown("Hi there! I'm here to help you with mental well-being. Ask me anything, upload a file, or speak to me. 🌿")

    chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, type="messages")

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_count="multiple",
        placeholder="Type your message or upload a file...",
        show_label=False,
        sources=["upload"],
    )

    audio_input = gr.Microphone(type="filepath", label="🎤 Speak here (optional)")

    chat_msg = chat_input.submit(add_message, [chatbot, chat_input, audio_input], [chatbot, chat_input])
    bot_msg = chat_msg.then(bot, chatbot, chatbot)
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

app.launch(share=True, debug=True)
