import os
import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.callbacks import get_openai_callback

# --- Secrets & Environment ---
open_ai_apikey = st.secrets["OPEN_AI_API_KEY"]
langchain_api_key = st.secrets.get("LANGCHAIN_API_KEY", "")
langchain_project = st.secrets.get("LANGCHAIN_PROJECT", "")

os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = langchain_project

# --- LLM Initialization ---
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=open_ai_apikey)

# --- Load PDF Knowledge Base ---
loader = WebBaseLoader("https://robkerrai.blob.core.windows.net/blogdocs/EDA_Cheat_Sheet.pdf?ref=robkerr.ai")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=open_ai_apikey)
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})  # Retrieve top document

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

# Preview uploaded CSV
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Here's a preview of your data:")
    st.dataframe(df.head())

# Chat input
input_text = st.chat_input("Ask anything about your uploaded data or EDA concepts:")

if input_text:
    with st.spinner("Generating response..."):

        if df is not None:
            # Create summary of the CSV
            df_summary = f"""This dataset has {df.shape[0]} rows and {df.shape[1]} columns.
Column names: {list(df.columns)}
Data types:\n{df.dtypes.to_string()}
Missing values:\n{df.isnull().sum().to_string()}"""
            context = [Document(page_content=df_summary)]
            st.write("CSV Summary:")
            st.code(df_summary)
        else:
            # Use PDF-based context if no CSV
            top_doc = retriever.invoke(input_text)[0]
            context = [top_doc]

        # Run chain with token tracking
        with get_openai_callback() as cb:
            response = chain.invoke({"context": context, "question": input_text})

        st.write("Answer:")
        st.markdown(response)

        # Token usage display
        st.write("Token Usage")
        st.write(f"Total tokens used: {cb.total_tokens}")
        st.write(f"Prompt tokens: {cb.prompt_tokens}")
        st.write(f"Completion tokens: {cb.completion_tokens}")
        st.write(f"Estimated cost (USD): ${cb.total_cost:.6f}")
