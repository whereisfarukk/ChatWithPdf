import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer
load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    model = SentenceTransformer('hkunlp/instructor-xl')
    embeddings = [model.encode(chunk) for chunk in text_chunks]
    vectorstore = FAISS.from_embeddings(embeddings)
    return vectorstore

    # sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
    # instruction = "Represent the Science title; Input:"
    # model = SentenceTransformer('hku-nlp/instructor-base')
    # embeddings = model.encode([[instruction,sentence,0]])
    # print(embeddings)

def main():
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")
    st.header("Chat with Multiple PDFs :books:")
    st.text_input("Ask a question about your document")
    
    with st.sidebar:
        st.subheader("Read your documents:")
        pdf_docs = st.file_uploader("Upload your PDF documents here", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs)
                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # Create vector store
                vectorstore = get_vectorstore(text_chunks)

if __name__ == '__main__':
    main()
