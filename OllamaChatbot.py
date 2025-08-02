import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

import os
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')

#Langsmith Tracking
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="O&A Chatbot with Ollama"

prompt=ChatPromptTemplate.from_messages([("system","You are a useful assistant please respond to the human queries"),
                           ("human","Question:{input}")])

def generate_response(question,llm,temperature,max_token):
   
    llm=ChatOllama(model=llm)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({"input":question})
    return answer
title=st.title("Enhanced Q&A Chatbot with OpenAI")

st.sidebar.title("Settings")
#Drop down to select various OpenAI models 
llm=st.sidebar.selectbox("Select an Open AI Model",["gemma3","llama3",'llama2'])

temperature=st.sidebar.slider("Temperatire",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

#Main UI for user Input

st.write("Go ahead and ask any question")
user_input=st.text_input("You:")

if user_input:
    response=generate_response(question=user_input,llm=llm,temperature=temperature,max_token=max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")