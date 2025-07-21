import os
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.agents import AgentType, AgentExecutor,initialize_agent
from langchain.agents import Tool
from langchain.chains import LLMChain, LLMMathChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks import StreamlitCallbackHandler
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader,YoutubeAudioLoader, YoutubeLoader, UnstructuredURLLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser

load_dotenv()


# -------------------- Streamlit UI Setup -------------------- #
st.set_page_config(page_title="Multifunctional Chatbot using Groq and Gemma 2")
st.title("🐦🔗 MultiFunctional Chatbot including RAG and Chat History using GROQ and Gemma 2")


st.sidebar.title("⚙️ Settings")

api_key=st.sidebar.text_input("Enter your Groq API Key:",type="password")

if not api_key:
    st.info("Please Enter your api key to continue!")
    st.stop()



# -------------------- Load LLM -------------------- #
llm=ChatGroq(model_name="gemma2-9b-it",groq_api_key=api_key)


# -------------------- Tools Setup -------------------- #
wiki=WikipediaAPIWrapper()
wiki_tool=Tool(
    name="Wikipedia",
    func=wiki.run,
    description="Tool for searching the web to find your necessary solutions. Only web based queries are allowed. Do not " \
                "use this for research papers or documents. Use the Arxiv tool for that purpose."
)

llm_math_chain=LLMMathChain(llm=llm)
calc=Tool(
    name="Calculator",
    func=llm_math_chain.run,
    description="Tool for solving your math based queries. Only math expressions are allowed."
)

arxiv=ArxivAPIWrapper()
arxiv_tool=Tool(
    name="Reads Research papers and documents",
    func=arxiv.run,
    description="Use only to read academic research papers from Arxiv.org, typically for technical or scientific topics. Do not use for general or entertainment-related questions.Do not use this tool if a local PDF is uploaded or if the question is about an uploaded document."
)

embeddings=OpenAIEmbeddings()

# -------------------- Document Upload + Retriever -------------------- #
uploaded_documents=st.file_uploader(label="Choose a PDF file",type="pdf",accept_multiple_files=True)

if uploaded_documents:
    document=[]
    for uploaded_file in uploaded_documents:
        os.makedirs("temp",exist_ok=True)
        pdf_path=os.path.join("temp",uploaded_file.name)
        with open(pdf_path,"wb") as file:
            file.write(uploaded_file.getbuffer())
        loaders=PyPDFLoader(pdf_path)
        document.extend(loaders.load())
    docs=document
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    fin_split=text_splitter.split_documents(docs)
    db=Chroma.from_documents(fin_split,embeddings)
    retriever=db.as_retriever()
else:
    retriever=None

# -------------------- Tool List -------------------- #
tools=[wiki_tool,calc,arxiv_tool]

def summarize_pdf(prompt, retriever, llm):
    docs = retriever.get_relevant_documents(prompt)
    if not docs:
        return "No relevant content found to summarize."
    
    combined_text = "\n\n".join([doc.page_content for doc in docs])
    merged_doc = [Document(page_content=combined_text)]
    
    summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")
    return summarize_chain.run(merged_doc)

if retriever:
    tools.append(Tool(
    name="Summarizer",
    func=lambda query: summarize_pdf(query, retriever, llm),
    description="Summarize the uploaded PDF's."
    ))
    tools.append(Tool(
        name="RAG model",
        func=lambda query: "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(query)]),
        description="Use this tool only for answering questions based on uploaded PDFs. Retrieve and explain contents from uploaded PDF files. Do not use this tool for research papers from arXiv or online academic sources."
    ))


# -------------------- Agent Setup -------------------- #
agent=initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# -------------------- Session + Chat History -------------------- #

session_id=st.text_input("Session ID:",value="default_session")

if "store" not in st.session_state:
        st.session_state.store={}

def get_session(session:str)->BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session]=ChatMessageHistory()
    return st.session_state.store[session]


if "messages" not in st.session_state:
    st.session_state["messages"]=[{"role":"assistant","content":"Hi! I am a multipurpose AI Chatbot. I can be a RAG model, Calculator, Summarizer and many more!."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# -------------------- Handle Chat Input -------------------- #
if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Get session-specific chat history
    history = get_session(session_id)
    history.add_user_message(prompt)

    # Handler for streaming response
    callback_handler = StreamlitCallbackHandler(st.container())

    try:
        inputs = {
            "input": prompt,
            "chat_history": history.messages,
        }
        # Check if prompt is about Arxiv
        if "arxiv" in prompt.lower() or "research paper" in prompt.lower() or "research papers" in prompt.lower():
            response = arxiv.run(prompt)
            
        elif retriever and ("document" in prompt.lower() or "pdf" in prompt.lower()):
        # You can also add more sophisticated checks here
          rag_answer = "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(prompt)])
          if rag_answer.strip():
              response = rag_answer
          else:
            # Fall back to agent if no match found in PDF
            response = agent.run(inputs, callbacks=[callback_handler])
        else:
            response = agent.run(inputs, callbacks=[callback_handler])
    except Exception as e:
        response = f"⚠️ Sorry, I encountered an error: {e}"
    
    history.add_ai_message(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

