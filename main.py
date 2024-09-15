import os
import tempfile
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory


st.title("🦜 LangChain: Chat with Documents")

# define azure openai endpoint and key
os.environ["OPENAI_API_VERSION"] = "2024-05-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = ""
os.environ["AZURE_OPENAI_API_KEY"] = ""

# retriever fuunvtion with streamlit cache decorator to run the 
# file reading and embedding only once 
@st.cache_resource
def configure_retriever(uploaded_files):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    # model name should be the deployment name in azure studio
    embeddings = AzureOpenAIEmbeddings(chunk_size=1, model="embedding")
    vectordb = FAISS.from_documents(splits, embeddings)

    # Define retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    return retriever


# file uploading logic in streamlit
uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    st.stop()

# configure retriever with uploaded files
retriever = configure_retriever(uploaded_files)

# chat history 
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, chat_memory=msgs)


# template formation
template = """You are an AI chatbot having a conversation with a human.
Use the following pieces of context to answer the question at the end
Whenever you think that there is no answer from the context, ask for clarification and further context.
{context}
History
{chat_history}  
Human: {question}
AI: """

prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# define LLM, model name should be deployment name in azure studio
llm = AzureChatOpenAI(model="gptmodel")

# conversational retriever from langchain, define the params needed
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type='stuff',
    retriever=retriever,
    memory=memory,
    verbose=True,
    combine_docs_chain_kwargs={
        'prompt': prompt,
    }
)

# display msgs
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# input query and return answer
if query := st.chat_input():
    st.chat_message("user").write(query)
    response = qa({"question": query})
    st.chat_message("assistant").write(response["answer"])
