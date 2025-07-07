import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import RunnableWithMessageHistory


# 📌 Load secrets
open_ai_apikey = st.secrets["OPEN_AI_API_KEY"]
langchain_api_key = st.secrets.get("LANGCHAIN_API_KEY", "")
langchain_project = st.secrets.get("LANGCHAIN_PROJECT", "")
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = langchain_project

# ✅ Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=open_ai_apikey)

# ✅ Load documents
loader = WebBaseLoader("https://robkerrai.blob.core.windows.net/blogdocs/EDA_Cheat_Sheet.pdf?ref=robkerr.ai")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=open_ai_apikey)
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# ✅ Prompt with history
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template="""You are a Python EDA expert. Write code based on the context below and use slang when appropriate. Use the chat history to guide your answers.

Chat History:
{history}

Context:
{context}

Question:
{question}

Answer:"""
)

# ✅ Chain
chain = create_stuff_documents_chain(llm, prompt)
parser = StrOutputParser()
retriever_chain = (
    RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough()
    }) | chain | parser
)

# ✅ Memory per session
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

memory = st.session_state.memory

retriever_chain_with_memory = RunnableWithMessageHistory(
    retriever_chain,
    lambda session_id: memory,  # Per-session memory
    input_messages_key="question",
    history_messages_key="history"
)

# ✅ Streamlit UI
st.title("📊 EDA Expert")
st.subheader("Ask anything about EDA in Python")

user_input = st.chat_input("Ask your question:")

if user_input:
    with st.spinner("Thinking..."):
        response = retriever_chain_with_memory.invoke(
            {"question": user_input},
            config={"configurable": {"session_id": "default"}}
        )
    st.write(response)

# ✅ Optional: Show chat history in sidebar
with st.sidebar:
    st.subheader("Chat History")
    for msg in memory.chat_memory.messages:
        st.markdown(f"**{msg.type.capitalize()}:** {msg.content}")
