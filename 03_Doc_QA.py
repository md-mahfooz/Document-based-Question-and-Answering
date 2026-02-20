import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableParallel, RunnablePassthrough
from pydantic import BaseModel, Field
from typing import Literal
from operator import itemgetter

st.title("Document based Q&A")
st.write("Upload a PDF, and I will answer questions, summarize it, or create a study guide!")


uploaded_file = st.file_uploader("Upload your PDF here", type="pdf")


if uploaded_file is not None:
    

    if "vectorstore" not in st.session_state:
        with st.spinner("Reading and analyzing your PDF..."):
            
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            
            st.session_state.vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            
            st.success("PDF loaded successfully! You can now ask questions.")

    
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
    
    model = ChatOllama(model="llama3", temperature=0)
    model_json = ChatOllama(model="llama3", format="json", temperature=0)
    string_parser = StrOutputParser()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    
    class QueryIntent(BaseModel):
        intent: Literal['qa', 'summary', 'study_guide'] = Field(description="Classify the user intent.")

    parser = PydanticOutputParser(pydantic_object=QueryIntent)

    classifier_prompt = PromptTemplate(
        template="""You are a strict routing assistant. Classify the user query.
        Rules:
        - If they ask a specific question -> "qa"
        - If they ONLY ask for a summary or notes -> "summary"
        - If they explicitly ask for a "study guide" or a "quiz" -> "study_guide"
        
        You MUST output valid JSON with exactly one key named "intent".
        
        Query: {question}\n{format_instructions}""",
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    classifier_chain = classifier_prompt | model_json | parser

    
    qa_prompt = PromptTemplate(
        template="Answer the question based ONLY on the context.\nContext: {context}\nQuestion: {question}",
        input_variables=['context', 'question']
    )
    qa_chain = (
        {"context": itemgetter("question") | retriever | format_docs, "question": itemgetter("question")}
        | qa_prompt | model | string_parser
    )

    summary_prompt = PromptTemplate(
        template="Generate short bullet-point notes summarizing this text:\n{context}", 
        input_variables=['context']
    )
    summary_chain = (
        {"context": itemgetter("question") | retriever | format_docs}
        | summary_prompt | model | string_parser
    )

    quiz_prompt = PromptTemplate(
        template="Generate 3 short questions and answers from this text:\n{context}", 
        input_variables=['context']
    )
    merge_prompt = PromptTemplate(
        template="Merge the notes and quiz into a clean document.\nNotes:\n{notes}\nQuiz:\n{quiz}",
        input_variables=['notes', 'quiz']
    )
    parallel_generation = RunnableParallel(
        {"notes": summary_prompt | model | string_parser, "quiz": quiz_prompt | model | string_parser}
    )
    study_guide_chain = (
        {"context": itemgetter("question") | retriever | format_docs}
        | parallel_generation | merge_prompt | model | string_parser
    )

   
    branch_logic = RunnableBranch(
        (lambda x: x["intent"].intent == 'summary', summary_chain),
        (lambda x: x["intent"].intent == 'study_guide', study_guide_chain),
        qa_chain  
    )

    final_chain = RunnablePassthrough.assign(intent=classifier_chain) | branch_logic

    
    user_question = st.text_input("What would you like to know or do with the PDF?")

    if st.button("Submit"):
        if user_question:
            with st.spinner("Processing your request..."):
                response = final_chain.invoke({"question": user_question})
                st.write(response)
        else:
            st.warning("Please type a question first.")
else:
    st.info("Please upload a PDF to get started!")