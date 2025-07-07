import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Streamlit Secrets (add these in your Streamlit Cloud app settings)
open_ai_apikey = st.secrets["OPEN_AI_API_KEY"]
langchain_api_key = st.secrets.get("LANGCHAIN_API_KEY", "")
langchain_project = st.secrets.get("LANGCHAIN_PROJECT", "")

# Optional: set LangChain tracking environment (only needed if you're using LangSmith)
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = langchain_project

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=open_ai_apikey)

# Load and process the web page
loader = WebBaseLoader("https://robkerrai.blob.core.windows.net/blogdocs/EDA_Cheat_Sheet.pdf?ref=robkerr.ai")
documents = loader.load()

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Create a vector store from the documents
embeddings = OpenAIEmbeddings(openai_api_key=open_ai_apikey)
vectorstore = FAISS.from_documents(docs, embeddings)

# Create a retriever chain
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a Python EDA expert and you should write code based on the provided context when the user asks you how to do a certain task. Talk the way GEN-Z people talk by using slang in appopriate areas and there is no need to use a formal tone.

Context: {context}

Question: {question}
Answer:"""
)

chain = create_stuff_documents_chain(llm, prompt=prompt)

retriever = vectorstore.as_retriever()
parser = StrOutputParser()

retriever_chain = (
    RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough(),
    })
    | chain | parser
)

# Streamlit UI
st.title("📊 EDA Expert")
st.subheader("Get help from this awesome friendly bot")
input_text = st.chat_input("Ask a question about EDA in Python:")
if input_text:
    with st.spinner("Cooking up some shit..."):
        response = retriever_chain.invoke(input_text)
    st.write(response)
