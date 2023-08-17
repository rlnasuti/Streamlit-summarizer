# External imports
import openai
import streamlit as st

# langchain imports
from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents import OpenAIFunctionsAgent
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent, create_retriever_tool
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.tools import tool
from langchain.document_loaders import PyPDFLoader

from dotenv import load_dotenv

load_dotenv()

# Constants
EMBEDDINGS_CHUNK_SIZE = 10

def setup_faiss_db():
    """Set up the FAISS db and create a retriever tool."""
    embeddings = OpenAIEmbeddings(chunk_size=EMBEDDINGS_CHUNK_SIZE)
    db = FAISS.load_local("./dbs/langchain/faiss_index", embeddings=embeddings)
    return db.as_retriever()

def create_summarize_chain():
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    chain = load_summarize_chain(llm, chain_type="refine")
    return chain

@tool
def summarize_document(pdf_path: str) -> str:
    """useful for summarizing a pdf file"""
    summarizer = create_summarize_chain()
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    summary = summarizer.run(docs)
    return summary


def setup_agent_executor(retriever, llm):
    """Set up the Conversational Retrieval Agent."""
    langchain_QA = create_retriever_tool(retriever, "search_langchain_repo", "Searches and returns documents from the Langchain git repository.")
    tools = [langchain_QA, summarize_document]
    memory_key = "history"
    memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)
    system_message = SystemMessage(
        content=(
            "Do your best to answer the questions. "
            "Feel free to use any tools available to look up "
            "relevant information, only if necessary"
        )
    )
    prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message, extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)])
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, return_intermediate_steps=True)

st.title("ChatGPT-like clone with FAISS-backed Retrieval using Conversational Retrieval Agent")

# File upload functionality for PDF in sidebar
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    with open("uploaded_pdf_file.pdf", "wb") as f:  # Save the file with a specific name (you can change this name if needed)
        f.write(uploaded_file.getvalue())
    st.sidebar.write("Uploaded file successfully saved to uploaded_pdf_file.pdf")


# Sidebar with Model Selection
model_selection = st.sidebar.selectbox('Model Selection', ['gpt-3.5-turbo', 'gpt-4'])
st.session_state["openai_model"] = model_selection


openai.api_key = st.secrets["OPENAI_API_KEY"]

# Ensure initial session state
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def initialize_agent_executor(model_name):
    retriever = setup_faiss_db()
    llm = ChatOpenAI(temperature=0, model=model_name)
    return setup_agent_executor(retriever, llm)

# This will ensure agent_executor is initialized only once
agent_executor = initialize_agent_executor(st.session_state["openai_model"])

# Update agent_executor's memory from the session state if it exists
if 'agent_executor_memory' in st.session_state:
    agent_executor.memory = st.session_state.agent_executor_memory

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        result = agent_executor({"input": prompt})
        full_response = result["output"]
        st.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.agent_executor_memory = agent_executor.memory  # Update the agent's memory in the session state
