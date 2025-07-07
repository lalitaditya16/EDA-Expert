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
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage

# Load API keys from Streamlit secrets
open_ai_apikey = st.secrets["OPEN_AI_API_KEY"]
langchain_api_key = st.secrets.get("LANGCHAIN_API_KEY", "")
langchain_project = st.secrets.get("LANGCHAIN_PROJECT", "")

# Optional LangChain environment setup
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = langchain_project

# Initialize OpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=open_ai_apikey)

# Load and process the PDF
loader = WebBaseLoader("https://robkerrai.blob.core.windows.net/blogdocs/EDA_Cheat_Sheet.pdf?ref=robkerr.ai")
documents = loader.load()

# Chunk the documents
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)

# Vector store
embeddings = OpenAIEmbeddings(openai_api_key=open_ai_apikey)
vectorstore = FAISS.from_documents(docs, embeddings)

# Memory setup
memory = ConversationBufferMemory(return_messages=True)

# Prompt template
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template="""
You are a Python EDA expert and you should write code based on the provided context when the user asks you how to do a certain task. Talk the way GEN-Z people talk by using slang in appropriate areas and there is no need to use a formal tone. You also have the functionality to use chat history to return relevant data.

Chat History: {history}

Context: {context}

Question: {question}

Answer:"""
)

# Retrieval & chaining
chain = create_stuff_documents_chain(llm, prompt=prompt)
retriever = vectorstore.as_retriever()
parser = StrOutputParser()

retriever_chain = (
    RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough(),
        
    }) | chain | parser
)

# UI
st.title("📊 EDA Expert")
st.subheader("Get help from this awesome friendly bot")
input_text = st.chat_input("Ask a question about EDA in Python:")

if input_text:
    memory.chat_memory.add_message(HumanMessage(content=input_text))
    history = memory.load_memory_variables({})["history"]

    with st.spinner("Cooking up some shit..."):
        relevant_docs = retriever.invoke(input_text)
        response = chain.invoke({
            "content":relevant_docs,
            "question": input_text,
            "history": history
        })

    memory.chat_memory.add_message(AIMessage(content=response))
    st.write(response)
