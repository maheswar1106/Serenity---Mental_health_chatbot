import io
import os
from gtts import gTTS
import playsound
import speech_recognition as sr
import fitz  
import streamlit as st
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
import logging
from dotenv import load_dotenv


load_dotenv()
logging.basicConfig(level=logging.DEBUG)


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


pdf_path = "medical_book.pdf'  
pdf_text = extract_text_from_pdf(pdf_path)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.create_documents(pdf_text.split("\n\n"))


if not documents:
    logging.error("No content was extracted from the PDF.")
    st.error("The PDF content could not be loaded. Please check the file.")


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(documents, embeddings)


test_query = "Sample query"
test_results = vector_store.similarity_search(test_query, k=5)
if not test_results:
    logging.warning("Vector store retrieval failed for the test query.")


llm = OllamaLLM(model="llama2", temperature=0.7)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Context: {context}
    Question: {question}
    Please provide a detailed response highlighting:
    1. Advantages
    2. Disadvantages
    """
)


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template}
)

def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=15)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            st.error("Could not understand the audio.")
            return None
        except sr.RequestError as e:
            st.error(f"Error with the recognition service: {e}")
            return None

def speak_response(text):
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save("response.mp3")
    playsound.playsound("response.mp3", True)
    os.remove("response.mp3")


def medical_chatbot(query):
    try:
        result = qa_chain({"question": query})
        logging.debug(f"QA Chain output: {result}")

        response = result.get("answer", "Sorry, I couldn't generate a response.")
        source_docs = result.get("source_documents", [])

        if "advantages" not in response.lower() and "disadvantages" not in response.lower():
            response += "\n\nFor further details, please consult a certified medical professional."

        return response, source_docs
    except Exception as e:
        logging.error(f"Error in `medical_chatbot`: {e}")
        st.error(f"Error processing the query: {e}")
        return "An error occurred while generating the response.", []

st.title("AARMA - Medical Information Assistant")
st.write("Retrieve information from the medical book with insights into advantages and disadvantages.")

query = None
text_query = st.text_input("Type your query:")
voice_input_active = st.button("Speak Your Query")

if voice_input_active:
    st.write("Listening for your voice input...")
    query = get_voice_input()
    if query:
        st.write(f"You said: {query}")
else:
    if text_query:
        query = text_query
        st.write(f"You wrote: {query}")

if query:
    try:
        with st.spinner("Processing your query..."):
            response, source_docs = medical_chatbot(query)
        st.write(f"Bot: {response}")
        st.subheader("Source Documents:")
        for doc in source_docs:
            st.write(f"Document: {doc.page_content[:500]}...")
        speak_response(response)
    except Exception as e:
        logging.error(f"Error during chatbot execution: {e}")
        st.error(f"An error occurred: {str(e)}")
