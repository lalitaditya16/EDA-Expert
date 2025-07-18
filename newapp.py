import os
import streamlit as st
import pandas as pd
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.callbacks import get_openai_callback
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# --- Environment ---
groq_api_key = st.secrets["GROQ_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# --- LLM (still using GPT-3.5) ---
llm = ChatGroq(model="gemma2-9b-it", api_key=groq_api_key)

# --- Hugging Face Embeddings ---
embeddings = HuggingFaceEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
   
)

# --- Load and Embed EDA Cheat Sheet PDF ---
loader = WebBaseLoader("https://robkerrai.blob.core.windows.net/blogdocs/EDA_Cheat_Sheet.pdf?ref=robkerr.ai")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# --- Prompt Template ---
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You're a Gen-Z Python EDA expert. Write helpful Python code that works on the uploaded CSV file's dataframe (named `df`). 
Explain your reasoning and show the output. Keep it chill, use slang where it fits.

Context: {context}

Question: {question}
Answer:"""
)

# --- Chain ---
chain = create_stuff_documents_chain(llm, prompt=prompt)
parser = StrOutputParser()

# --- Streamlit UI ---
st.title("EDA Expert")
st.subheader("Upload a CSV and ask questions")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
df = None

if input_text:
    with st.spinner("Generating response..."):
        if df is not None:
            df_summary = f"""This dataset has {df.shape[0]} rows and {df.shape[1]} columns.
Column names: {list(df.columns)}
Data types:\n{df.dtypes.to_string()}
Missing values:\n{df.isnull().sum().to_string()}"""
            context = [Document(page_content=df_summary)]
            st.write("CSV Summary:")
            st.code(df_summary)
        else:
            top_doc = retriever.invoke(input_text)[0]
            context = [top_doc]

        #  Invoke the chain and show the response
        with get_openai_callback() as cb:
            response = chain.invoke({"context": context, "question": input_text})

        st.write("### Answer:")
        st.markdown(response)

        st.write("### Token Usage")
        st.write(f"Total tokens used: {cb.total_tokens}")
        st.write(f"Prompt tokens: {cb.prompt_tokens}")
        st.write(f"Completion tokens: {cb.completion_tokens}")
        st.write(f"Estimated cost (USD): ${cb.total_cost:.6f}")
