import streamlit as st
from haystack import Pipeline, Document
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, PromptNode, PromptTemplate, PromptModel, FARMReader
from haystack.nodes import Docs2Answers

from haystack.nodes.retriever.web import WebRetriever
from transformers import AutoModelForCausalLM, AutoTokenizer  
import json, re
from pathlib import Path


def clean_content(data):
    cleaned_data = []
    for item in data:
        content = item.get('content', '')

        # Remove '\n' characters
        content = content.replace('\n', ' ')

        # Remove links (anything that looks like a URL)
        content = re.sub(r'http\S+', '', content)  # Remove http and https URLs
        content = re.sub(r'http\S+|www\.\S+|ID: \S+', '', content)  # Remove www URLs

        # Remove ID patterns like 'ID: ...'
        content = re.sub(r'ID: \S+', '', content)
        content = content.replace('All rights reserved.', '')
        # Update the content in the dictionary
        item['content'] = content
        cleaned_data.append(item)

    return cleaned_data
    

@st.cache_resource(show_spinner=False)
def get_plain_pipeline():

    plain_llm_template = PromptTemplate("Answer the following question: {query}")
    node = PromptNode(model_name_or_path = "google/flan-t5-base", default_prompt_template=plain_llm_template, max_length=300)

    pipeline = Pipeline()

    pipeline.add_node(component=node, name="prompt_node", inputs=["Query"])
    return pipeline

# @st.cache_resource(show_spinner=False)
# def get_plain_pipeline():
#     prompt_open_ai = PromptModel(model_name_or_path="text-davinci-003", api_key=st.secrets["OPENAI_API_KEY"])
#     # Now let make one PromptNode use the default model and the other one the OpenAI model:
#     plain_llm_template = PromptTemplate(name="plain_llm", prompt_text="Answer the following question: {query}")
#     node_openai = PromptNode(prompt_open_ai, default_prompt_template=plain_llm_template, max_length=300)
#     pipeline = Pipeline()
#     pipeline.add_node(component=node_openai, name="prompt_node", inputs=["Query"])
#     return pipeline



@st.cache_resource(show_spinner=False)
def get_retrieval_augmented_pipeline():
    file_path = r"data.json"
    with Path(file_path).open(encoding='utf-8') as file:
        data = json.load(file)

        reformat_data =  [{'Source Link': item['href'], 'content': item['text']} for item in data if 'text' in item]
        cleaned_data = clean_content(reformat_data)
        # # convert to Haystack Documents
        documents = [Document.from_dict(doc) for doc in cleaned_data]

        # set up the Document Store
        document_store = InMemoryDocumentStore(use_bm25=True)
        # write the documents to the Document Store
        document_store.write_documents(documents)

        # retrieve all documents from your document store
        all_documents = document_store.get_all_documents()
        combined_text = " ".join([doc.content for doc in all_documents])

        # set up retriever
        # BM25Retriever used by search algorithms to rank documents based on their relevance to a given search query.
        retriever = BM25Retriever(document_store=document_store)

        model = AutoModelForCausalLM.from_pretrained(
            'mosaicml/mpt-7b-instruct',
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        prompt_node = PromptNode("mosaicml/mpt-7b-instruct", model_kwargs={"model":model, "tokenizer": tokenizer})

        # build the Pipeline
        pipeline = Pipeline()
        pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])
        pipeline.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])

    return pipeline, document_store

@st.cache_resource(show_spinner=False)
def get_web_retrieval_augmented_pipeline():
    search_key = st.secrets["WEBRET_API_KEY"]
    web_retriever = WebRetriever(api_key=search_key, search_engine_provider="SerperDev")
    
    default_template = PromptTemplate("Given the context please answer the question. Context: {join(documents)}; Question: "
                    "{query}; Answer:",
    )


    # initiate the PromptNode
    node = PromptNode(model_name_or_path = "google/flan-t5-base", default_prompt_template=default_template,
                      max_length=500)

    # create a pipeline with the webretriever + PromptNode
    pipeline = Pipeline()
    pipeline.add_node(component=web_retriever, name='retriever', inputs=['Query'])
    pipeline.add_node(component=node, name="prompt_node", inputs=["retriever"])
    return pipeline
