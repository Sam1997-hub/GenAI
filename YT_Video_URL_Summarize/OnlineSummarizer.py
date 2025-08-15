import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import yt_dlp
import requests
import xml.etree.ElementTree as ET


# ---------- Custom YouTube Transcript Loader ----------
def load_youtube_transcript(url: str):
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": True,
        "subtitleslangs": ["en"],
        "writeautomaticsub": True,
    }

    transcript_text = ""

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        subs = info_dict.get("subtitles") or info_dict.get("automatic_captions")
        if subs and "en" in subs:
            sub_url = subs["en"][0]["url"]
            resp = requests.get(sub_url)

            # Try parsing XML captions
            try:
                root = ET.fromstring(resp.text)
                transcript_text = " ".join([t.text for t in root.findall(".//text") if t.text])
            except:
                transcript_text = resp.text  # fallback raw

        else:
            transcript_text = "No subtitles found for this video."

    return [Document(page_content=transcript_text, metadata={"source": url})]


# ---------- Split Documents into Chunks ----------
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_documents(docs)


# ---------- Streamlit UI ----------
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

# Sidebar for API key
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

# Main URL input
generic_url = st.text_input("URL", label_visibility="collapsed")

# LLM setup
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Prompt templates
prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

refine_prompt = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template="""Refine the existing summary with the new content provided.
Keep it concise, include any important missing points, and preserve accuracy.
Existing summary:
{existing_answer}

New text:
{text}

Refined summary:"""
)

# ---------- Main Button Action ----------
if st.button("Summarize the Content from YT or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video or a website URL")
    else:
        try:
            with st.spinner("Processing..."):
                # Load docs depending on source
                if "youtube.com" in generic_url:
                    docs = load_youtube_transcript(generic_url)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": (
                                "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/116.0.0.0 Safari/537.36"
                            )
                        },
                    )
                    docs = loader.load()

                # Split into chunks for refine
                docs = chunk_documents(docs)

                # Summarization
                chain = load_summarize_chain(
                    llm,
                    chain_type="refine",
                    question_prompt=prompt,
                    refine_prompt=refine_prompt
                )
                output_summary = chain.run(docs)

                st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception: {e}")
