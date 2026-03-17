import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

pdf_dir = Path("../data/pdf_files")
pdf_files = list(pdf_dir.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError("No PDF files found in the specified directory.")

loader = PyPDFLoader(pdf_files[0])
doc = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(doc)

embeddings = OpenAIEmbeddings() # api_key automatically inferred from env var `OPENAI_API_KEY` if not provided.
vectorstore = FAISS.from_documents(documents=documents, embeddings=embeddings)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory)

query = "What are Shri's top 5 skills?"
results = qa.invoke({"question": query})
print(results["answer"])