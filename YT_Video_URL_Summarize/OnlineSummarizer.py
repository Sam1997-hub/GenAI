import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
import yt_dlp
from langchain.schema import Document
import requests


# ---------- Custom YouTube Transcript Loader ----------
def load_youtube_transcript(url: str):
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": True,
        "subtitleslangs": ["en"],  # Change if you want a different language
        "writeautomaticsub": True,  # Use auto-generated if official subs missing
    }

    transcript_text = ""

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        # Try to get subtitles (auto-generated or official)
        subs = info_dict.get("subtitles") or info_dict.get("automatic_captions")
        if subs and "en" in subs:
            sub_url = subs["en"][0]["url"]
            resp = requests.get(sub_url)
            transcript_text = resp.text
        else:
            transcript_text = "No subtitles found for this video."

    return [Document(page_content=transcript_text, metadata={"source": url})]


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

prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# ---------- Main Button Action ----------
if st.button("Summarize the Content from YT or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video or a website URL")
    else:
        try:
            with st.spinner("Waiting..."):
                # Load docs depending on source
                if "youtube.com" in generic_url:
                    docs = load_youtube_transcript(generic_url)  # already a list of Document
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

                # Summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")

