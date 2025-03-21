import json
import os
import sys
import boto3
import streamlit as st

from dotenv import load_dotenv
load_dotenv()


## We will be using Titan Embeddings Model To generate Embedding
#from langchain_aws import BedrockEmbeddings
from langchain_aws.embeddings import BedrockEmbeddings

#from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader



# Vector Embedding And Vector Store

from langchain.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


bedrock=boto3.client(service_name="bedrock-runtime",
                     region_name="us-east-1",
                     aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                     aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
                    aws_session_token=os.environ['AWS_SESSION_TOKEN'])


bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",client=bedrock)


## Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader(path = "data")
    documents = loader.load()

    # - in our testing Character split works better with this PDF data set
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,
                                                   chunk_overlap=1000)

    docs = text_splitter.split_documents(documents)
    return docs


def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")


def get_titan_llm():
    ##create the Anthropic Model
    model_id = "amazon.titan-text-premier-v1:0"
    llm = Bedrock(model_id=model_id, client=bedrock)
                  #model_kwargs={'maxTokens': 200})

    return llm


def get_llama2_llm():
    ##create the Anthropic Model
    model_id = "meta.llama3-8b-instruct-v1:0"
    llm = Bedrock(model_id=model_id, client=bedrock,
                  model_kwargs={'max_gen_len': 512,
                                "temperature": 0.5,
                                "top_p": 0.9
                                })

    return llm



prompt_template = """

Human: Use the following pieces of context to provide a
concise answer to the question at the end but use at least summarize with
250 words with detailed explanations. If you don't know the answer,
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_response_llm(llm,vectorstore_faiss,query):
    chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=chain({"query":query})
    return answer['result']


def main():
    st.set_page_config("Chat PDF")

    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Amazon Titan Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings,
                                           allow_dangerous_deserialization=True)
            llm = get_titan_llm()

            # faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings,
                                           allow_dangerous_deserialization=True)
            llm = get_llama2_llm()

            # faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

if __name__ == "__main__":
    main()