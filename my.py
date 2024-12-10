import io
from gtts import gTTS
import playsound
import os
import speech_recognition as sr
import pandas as pd
import streamlit as st
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from operator import itemgetter
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.DEBUG)


data = pd.read_csv('data_to_be_cleansed.csv')
target_mapping = {
    0: "Stress",
    1: "Depression",
    2: "Bipolar disorder",
    3: "Personality disorder",
    4: "Anxiety"
}
data['target_label'] = data['target'].map(target_mapping)
documents = data.apply(lambda row: f"{row['text']}\nCategory: {row['target_label']}", axis=1).tolist()

def generate_tokens(s):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_text(s)
    return text_splitter.create_documents(splits)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(
    documents,
    embeddings,
    metadatas=[{"id": str(i)} for i in range(len(documents))]
)

llm = OllamaLLM(model="llama2", temperature=0.7)

def reciprocal_rank_fusion(results, k=60):
    fused_scores = {}
    for rank, doc in enumerate(results):
        doc_id = doc.metadata['id']
        if doc_id not in fused_scores:
            fused_scores[doc_id] = 0
        fused_scores[doc_id] += 1 / (rank + k)
    return sorted(fused_scores.items(), key=itemgetter(1), reverse=True)

def generate_queries(query):
    return [query]

def rag_fusion_retriever(query, top_k=20):
    queries = generate_queries(query)
    all_docs = []
    for q in queries:
        docs = vector_store.similarity_search(q, k=top_k)
        all_docs.extend(docs)

    fused_results = reciprocal_rank_fusion(all_docs)
    final_docs = [
        doc for doc_id, _ in fused_results[:top_k]
        for doc in all_docs if doc.metadata['id'] == doc_id
    ]
    return final_docs


class RAGFusionRetriever(BaseRetriever):
    def get_relevant_documents(self, query):
        return rag_fusion_retriever(query)


retriever = RAGFusionRetriever()

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a mental health support assistant. 
    Context: {context}
    Based on the context above, provide an empathetic, informative response to the following question: {question}. 
    Additionally, suggest possible solutions or actions that can help address the user's concerns effectively.
    """
)


qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
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
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save("response.mp3")
    playsound.playsound("response.mp3", True)
    os.remove("response.mp3")


def save_source_documents(source_docs, folder_name="ghji"):
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    for i, doc in enumerate(source_docs):
        file_path = os.path.join(folder_name, f"source_doc_{i+1}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc.page_content)


def medical_chatbot(query):
    result = qa_chain({"query": query})
    response = result['result']
    source_docs = result['source_documents']
    
   
    save_source_documents(source_docs)

    if "possible solutions" not in response.lower() and "action" not in response.lower():
        response += "\n\nPlease consider discussing these concerns with a certified mental health professional for tailored advice."
    return response, source_docs

st.title("ARMA - Mental Health Assistant")
st.write("Interact with the assistant using your voice or type your query below.")

input_mode = st.radio("Select Input Mode:", ("Text Input", "Voice Input"))

if input_mode == "Text Input":
    query = st.text_input("Your Query:")
    if query:
        st.write(f"You said: {query}")
        if query.lower() == "stop":
            st.write("Ending session. Goodbye!")
        else:
            try:
                response, source_docs = medical_chatbot(query)
                st.write(f"Bot: {response}")
                st.subheader("Source Documents:")
                for doc in source_docs:
                    st.write(f"Document: {doc.page_content[:500]}...")
                speak_response(response)
            except Exception as e:
                logging.error(f"Error during chatbot execution: {e}")
                st.error(f"An error occurred: {str(e)}")

elif input_mode == "Voice Input":
    if st.button("Start Voice Interaction"):
        query = get_voice_input()
        if query:
            st.write(f"You said: {query}")
            if query.lower() == "stop":
                st.write("Ending session. Goodbye!")
            else:
                try:
                    response, source_docs = medical_chatbot(query)
                    st.write(f"Bot: {response}")
                    st.subheader("Source Documents:")
                    for doc in source_docs:
                        st.write(f"Document: {doc.page_content[:500]}...")
                    speak_response(response)
                except Exception as e:
                    logging.error(f"Error during chatbot execution: {e}")
                    st.error(f"An error occurred: {str(e)}")
