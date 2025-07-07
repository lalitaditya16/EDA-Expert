import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st


# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT']=os.getenv('LANGCHAIN_PROJECT')
# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4o")
# Load and process the web page
loader = WebBaseLoader("https://robkerrai.blob.core.windows.net/blogdocs/EDA_Cheat_Sheet.pdf?ref=robkerr.ai")
documents = loader.load()
# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
# Create a vector store from the documents
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
# Create a retriever chain

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a Python EDA expert and you should write  code based on the provided context when the user asks you how to do a certaian task.

Context: {context}

Question: {question}
Answer:"""
)
chain = create_stuff_documents_chain(llm, prompt=prompt)

retriever = vectorstore.as_retriever()
parser = StrOutputParser()
retriever_chain = (
    RunnableParallel({
        "context": retriever,                      # retriever gets passed the input query
        "question": RunnablePassthrough(),         # same query passed to {question}
    }) 
    | chain | parser                               # final prompt uses context + question
)   
input_text = st.chat_input("Ask a question about EDA in Python:")
if input_text:
    with st.spinner("Thinking..."):
        response = retriever_chain.invoke(input_text)
    st.write(response)
