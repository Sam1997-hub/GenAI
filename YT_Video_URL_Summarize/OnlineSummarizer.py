import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
import yt_dlp
from langchain.schema import Document

def load_youtube_transcript(url: str):
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": True,
        "subtitleslangs": ["en"],  # change to 'en' or desired language
        "writeautomaticsub": True,  # use auto-generated if official subs missing
    }

    transcript_text = ""

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        # Try to get subtitles (auto-generated or official)
        subs = info_dict.get("subtitles") or info_dict.get("automatic_captions")
        if subs and "en" in subs:
            sub_url = subs["en"][0]["url"]
            import requests
            resp = requests.get(sub_url)
            transcript_text = resp.text
        else:
            transcript_text = "No subtitles found for this video."

    return [Document(page_content=transcript_text, metadata={"source": url})]


## sstreamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')



## Get the Groq API Key and url(YT or website)to be summarized
with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",value="",type="password")

generic_url=st.text_input("URL",label_visibility="collapsed")

## Gemma Model USsing Groq API
llm =ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

prompt_template="""
Provide a summary of the following content in 300 words:
Content:{text}

"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video utl or website url")

    else:
        try:
            with st.spinner("Waiting..."):
                ## loading the website or yt video data
                if "youtube.com" in generic_url:
                    loader=load_youtube_transcript(generic_url)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs=loader.load()

                ## Chain For Summarization
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                output_summary=chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")

                    
