'''
This project is a research tool that process URLs using Langchain UnstructuredURLLoader
URLs are processed using Langchain UnstructuredURLLoader and the content from that URLs
is divided into chunks using RecursiveCharacterTextSplitter and that chunks are converted 
into vector embeddings and that vector embeddings are stored in the vector store using 
ChromaDB.
Everything is set and then finally we RetrievalQAWithSourcesChain initialize the chain
that takes our input query and find the relevant chunks from our vector store and give us the 
response

Tools and Technologies Used : 
1) Python 
2) Langchain modules 
3) OpenAI 
4) ChromaDB
5) RetrievalQAWithSourcesChain

'''



import os
import streamlit as st
import time
from dotenv import load_dotenv

from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Load environment variables (like OpenAI API key)
load_dotenv()

# Streamlit UI
st.title("Blink Research : Get quick insights📈")
st.sidebar.title("Add Your URLs to ")

# Input URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

# Language Model Setup
llm = OpenAI(temperature=0.9, max_tokens=500)

# Global variable for vectorstore
vectorstore = None

# When user clicks the process button
if process_url_clicked and urls:
    try:
        main_placeholder.text("🔄 Loading and processing URLs...")

        # Load content from URLs
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000,
            chunk_overlap=100
        )
        docs = text_splitter.split_documents(data)

        # ✅ Filter out empty chunks to avoid ChromaDB errors
        docs = [doc for doc in docs if doc.page_content.strip()]

        if not docs:
            st.error("⚠️ No valid content found in the provided URLs. Please check the input.")
        else:
            # Embedding with OpenAI
            embeddings = OpenAIEmbeddings()

            # Load into Chroma vector DB (in-memory)
            vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

            main_placeholder.success("✅ Processing complete! You can now ask questions.")

    except Exception as e:
        st.error(f"❌ Error while processing: {e}")

# Input box to ask questions
query = main_placeholder.text_input("Ask a question based on the articles:")

# Run the QA chain if everything is ready
if query:
    if vectorstore:
        try:
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )
            result = chain({"question": query}, return_only_outputs=True)

            st.header("Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                for source in sources.split("\n"):
                    st.write(source)
        except Exception as e:
            st.error(f"❌ Error during question answering: {e}")
    else:
        st.warning("⚠️ Please process the URLs first before asking questions.")
