from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import streamlit as st
from streamlit_option_menu import option_menu

def load_documents(directory_path):
    '''
    Loads pdf files from a directory and creates a list of Langchain documents
    '''
    loader = PyPDFDirectoryLoader(directory_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)
    final_documents = text_splitter.split_documents(documents)
    for document in final_documents:
        document.page_content = document.page_content.replace('\n', ' ')
    return final_documents

def initialize_models():
    '''
    Initialize embedding model and large language model
    '''
    embed_model = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5", 
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True} 
    )
    llm = ChatGroq(
        temperature=0,
        model="mixtral-8x7b-32768",
        api_key="gsk_****"
    )
    return embed_model, llm

def hybrid_retrieval(keyword_retriever, vector_retriever):
    '''
    Performs hybrid search (keyword + vector search)
    '''
    ensemble_retriever = EnsembleRetriever(retrievers=[keyword_retriever, vector_retriever], weights=[0.4, 0.6])
    return ensemble_retriever

def create_prompt():
    '''
    Define a prompt template for the Q&A system
    '''
    
    prompt_template = """
    You are an extremely intelligent and helpful tutor. 
    Strictly answer the given question only based on the following context

    {context}
    Question: {question}

    Helpful Answers:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return prompt

def main():
    st.set_page_config(page_title="LLM Whisperer", layout="wide")
    
    with st.sidebar:
        selected = option_menu(
            menu_title="",
            options=["Home", "Q&A"],
            icons=["house", "question-circle"],
            default_index=0,
        )

    if selected == "Home":
        st.title("Stay updated about recent advancements in LLMs")
        st.write("Shoot your questions in the 'Q&A' section!\n\n")
        st.write("""I'll be able to address nuanced queries specialized in the domain of Large Language Models (LLMs)""")

    if selected == "Q&A":
        st.title("Ask a Question")
        
        if "documents_loaded" not in st.session_state:
            st.session_state.documents_loaded = False

        if not st.session_state.documents_loaded:
            with st.spinner('Loading documents...'):
                directory_path = "C:\\Users\\Aditya\\Downloads\\llm_papers"
                final_documents = load_documents(directory_path)

                embed_model, llm = initialize_models()

                bm25_retriever = BM25Retriever.from_documents(final_documents)
                bm25_retriever.k = 3

                #load previously created vector index
                vector_store = FAISS.load_local('C:\\Users\\Aditya\\OneDrive\\llm_qa\\faiss_index_llms', embed_model, allow_dangerous_deserialization=True)
                vector_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

                ensemble_retriever = hybrid_retrieval(bm25_retriever, vector_retriever)

                prompt = create_prompt()

                retrievalQA = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=ensemble_retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": prompt}
                )
                st.session_state.retrievalQA = retrievalQA
                st.session_state.documents_loaded = True

        if st.session_state.documents_loaded:
            query = st.text_input("")
            if query:
                result = st.session_state.retrievalQA.invoke({"query": query})
                st.write("Answer:", result['result'])

if __name__ == "__main__":
    main()
