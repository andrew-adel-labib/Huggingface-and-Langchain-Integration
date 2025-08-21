import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

st.set_page_config(page_title="Summarize Text from Youtube or Website", page_icon="🦜")
st.title("🦜 Summarize Text from Youtube or Website")
st.subheader("Summarize URL Content")

with st.sidebar:
    hf_api_token = st.text_input(label="Huggingface API Token", value="", type="password")

url = st.text_input(label="URL", label_visibility="visible")

stuff_prompt_template = """
Provide a summary of the follwing content in 300 words:
Content: {text}
"""

stuff_prompt = PromptTemplate(template=stuff_prompt_template, input_variables=["text"])

if st.button("Summarize the content from Youtube or Website"):
    if not hf_api_token.strip() or not url.strip():
        st.error("Please, Provide the information to get started")
    elif not validators.url(url):
        st.error("Please, Enter a valid URL. It can be a Youtube Video URL or Website URL.")

    else:
        try:
            with st.spinner("waiting...."):
                repo_id = "google/gemma-2-9b"
                llm = HuggingFaceEndpoint(repo_id=repo_id,
                          huggingfacehub_api_token=hf_api_token,
                          temperature=0.7,
                          max_new_tokens=150)

                if "youtube.com" in url or "youtu.be" in url:
                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                else: 
                    loader = UnstructuredURLLoader(urls=[url], ssl_verify=False, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                
                docs = loader.load()

                if not docs:
                    st.error("No content could be extracted from this URL.")
                    st.stop()

                stuff_chain = load_summarize_chain(llm=llm,
                                                   chain_type="stuff",
                                                   prompt=stuff_prompt,
                                                   verbose=True)
                
                stuff_summary = stuff_chain.run(docs)

                st.success(stuff_summary)

        except Exception as e:
            st.exception(f"Exception: {e}")
           
          



