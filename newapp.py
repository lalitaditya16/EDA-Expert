import os
import streamlit as st
import uuid

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableWithMessageHistory
from langchain.memory.chat_message_histories import ChatMessageHistory


# ------------------------
# 🔐 Load API Keys from Streamlit Secrets
# ------------------------
open_ai_apikey = st.secrets["OPEN_AI_API_KEY"]
langchain_api_key = st.secrets.get("LANGCHAIN_API_KEY", "")
langchain_project = st.secrets.get("LANGCHAIN_PROJECT", "")

os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = langchain_project

# ------------------------
# 🤖 Initialize LLM
# ------------------------
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=open_ai_apikey)

# ------------------------
# 📄 Load and Process PDF
# ------------------------
loader = WebBaseLoader("https://robkerrai.blob.core.windows.net/blogdocs/EDA_Cheat_Sheet.pdf?ref=robkerr.ai")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# ------------------------
# 🧠 Create Vector Store
# ------------------------
embeddings = OpenAIEmbeddings(openai_api_key=open_ai_apikey)
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# ------------------------
# 📋 Prompt Template
# ------------------------
prompt = PromptTemplate(
    input_variables=["context", "question", "history"],
    template="""
You're a GEN-Z style Python EDA expert. Based on the following context and chat history, answer the user's question in a friendly, slightly slangy tone.

Chat History:
{history}

Context:
{context}

Question:
{question}

Answer:"""
)

# ------------------------
# 🔗 Create Chain
# ------------------------
chain = create_stuff_documents_chain(llm, prompt=prompt)

retriever_chain = (
    RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough(),
    })
    | chain
    | StrOutputParser()
)

# ------------------------
# 🧠 Set Up Per-Session Memory
# ------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

retriever_chain_with_memory = RunnableWithMessageHistory(
    retriever_chain,
    lambda session_id: st.session_state.chat_history,
    input_messages_key="question",
    history_messages_key="history"
)

# ------------------------
# 💬 Streamlit UI
# ------------------------
st.title("📊 EDA Expert")
st.subheader("Ask anything about EDA in Python")

user_input = st.chat_input("Ask your question:")

if user_input:
    with st.spinner("Thinking hard..."):
        response = retriever_chain_with_memory.invoke(
            {"question": user_input},
            config={"configurable": {"session_id": st.session_state.session_id}}
        )
    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(response)
