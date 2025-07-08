import os
import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- Setup API Keys ---
open_ai_apikey = st.secrets["OPEN_AI_API_KEY"]
langchain_api_key = st.secrets.get("LANGCHAIN_API_KEY", "")
langchain_project = st.secrets.get("LANGCHAIN_PROJECT", "")

os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = langchain_project

# --- Load LLM ---
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=open_ai_apikey)

# --- PDF-based Vector Context ---
loader = WebBaseLoader("https://robkerrai.blob.core.windows.net/blogdocs/EDA_Cheat_Sheet.pdf?ref=robkerr.ai")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings(openai_api_key=open_ai_apikey)
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# --- Streamlit UI ---
st.title("📊 EDA Expert")
st.subheader("Upload a CSV and ask questions!")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])
df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("🔍 Here's a preview of your data:")
    st.dataframe(df.head())

# --- Prompt ---
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You're a Gen-Z Python EDA expert. Write helpful Python code based on context.
Use slang casually. Context is either from an uploaded dataset or EDA cheat sheet.

Context: {context}

Question: {question}
Answer:"""
)

chain = create_stuff_documents_chain(llm, prompt=prompt)
parser = StrOutputParser()

# --- Chat Input ---
input_text = st.chat_input("Ask anything about your uploaded data or EDA concepts:")

if input_text:
    with st.spinner("Thinking hard 💡..."):
        # Generate CSV summary if available
        if df is not None:
            df_summary = f"""This dataset has {df.shape[0]} rows and {df.shape[1]} columns.
Column names: {list(df.columns)}
Data types:\n{df.dtypes.to_string()}
Missing values:\n{df.isnull().sum().to_string()}"""
            context = [Document(page_content=df_summary)]
        else:
            # Use PDF as fallback context
            context = retriever.invoke(input_text)[0].page_content

        response = chain.invoke({"context": context, "question": input_text})
        st.write(response)
