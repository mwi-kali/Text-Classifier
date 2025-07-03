import logging


import streamlit as st


from concurrent.futures import ThreadPoolExecutor
from text_classifier.classification import Classifier
from text_classifier.config import MAX_WORKERS
from text_classifier.ingestion import Ingestor
from text_classifier.schemas import ClassificationResult
from text_classifier.summarization import Summarizer
from text_classifier.utils import configure_logging, is_url, preprocess_text


st.set_page_config(
    page_title="Text Classifier & Summarizer",
    page_icon="📝",
    layout="wide",
)

configure_logging()

st.markdown(
    """
    <style>
      body { background-color: #111827; color: #f9fafb; }
      .block-container { padding: 1rem 2rem; max-width: 1400px; }
      h1,h2,h3 { font-family:'Segoe UI',sans-serif; color: #f9fafb; }
      .stSidebar { background-color: #1f2937}
      .metric-card {
        background-color: #1f2937;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        text-align: center;
      }
      .metric-card h3 {
        margin: 0;
        color: #9ca3af;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }
      .metric-card p {
        margin: 0.25rem 0 0;
        font-size: 1.4rem;
        font-weight: bold;
      }
      .topic-badge {
        display: inline-block;
        background-color: #374151;
        color: #f9fafb;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.9rem;
        margin: 2px 4px;
        font-size: 0.85rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Classifier & Summarizer")
st.write(
    """
    Upload a PDF/TXT file or enter a web/YouTube URL to extract, preprocess, classify, and summarize your content.
    """
)

st.sidebar.header("Ingestion")
source_type = st.sidebar.radio("Pick a Source Type", ["File", "URL"])
uploaded_file = (
    st.sidebar.file_uploader("Upload a .PDF or .TXT file", type=["pdf", "txt"])
    if source_type == "File"
    else None
)
url_input = (
    st.sidebar.text_input("Enter Web or YouTube URL")
    if source_type == "URL"
    else None
)

st.sidebar.markdown("---")
st.sidebar.header("Settings")
run_cls = st.sidebar.checkbox("Run Classification", value=True)
run_sum = st.sidebar.checkbox("Run Summarization", value=True)
sum_mode = st.sidebar.radio("Summarization Mode", ["Abstractive", "Map-Reduce"])

if not st.sidebar.button("▶️ Process", use_container_width=True):
    st.stop()


ingestor = Ingestor()
try:
    if source_type == "File":
        if not uploaded_file:
            st.error("Please upload a .PDF or .TXT file.")
            st.error("Btw... It is in the sidebar.")
            st.stop()
        tmp_path = f"/tmp/{uploaded_file.name}"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        raw = ingestor.ingest(tmp_path, is_url=False)
    else:
        if not url_input or not is_url(url_input):
            st.error("Please enter a valid URL.")
            st.error("Btw... It is in the sidebar. When you’re ready, press Enter to finalize and lock in your selection.")
            st.stop()
        raw = ingestor.ingest(url_input, is_url=True)

    text = preprocess_text(raw, lower=False, use_spacy=False)
    if not text:
        st.error("Failed to extract any text from the source.")
        st.stop()

except Exception as e:
    st.error(f"Ingestion error: {e}")
    st.stop()

classifier = Classifier()
summarizer = Summarizer()
executor = ThreadPoolExecutor(max_workers=2)

cls_fut = (
    executor.submit(classifier.classify_all, text)
    if run_cls
    else None
)

if run_sum:
    if sum_mode == "Map-Reduce":
        sum_fut = executor.submit(summarizer.summarize_map_reduce, text)
    else:
        sum_fut = executor.submit(
            summarizer.abstractive_summary, text        )
else:
    sum_fut = None

executor.shutdown(wait=False)


st.header("Classification")
if not run_cls:
    st.info("Classification is turned off.")
else:
    cls_data = cls_fut.result()
    cls = ClassificationResult(**cls_data)
    s = cls.sentiment.lower()
    if "neg" in s:
        col = "#ef4444"
    elif "pos" in s:
        col = "#10b981"
    else:
        col = "#9ca3af"

    r1 = st.columns(3)
    r2 = st.columns(2)

    with r1[0]:
        st.markdown(
            f"<div class='metric-card'><h3>Sentiment</h3>"
            f"<p style='color:{col}'>{cls.sentiment.title()}</p></div>",
            unsafe_allow_html=True,
        )
    with r1[1]:
        st.markdown(
            f"<div class='metric-card'><h3>Language</h3>"
            f"<p>{cls.language.upper()}</p></div>",
            unsafe_allow_html=True,
        )
    with r1[2]:
        st.markdown(
            f"<div class='metric-card'><h3>Style</h3>"
            f"<p>{cls.style.title()}</p></div>",
            unsafe_allow_html=True,
        )

    with r2[0]:
        st.markdown(
            f"<div class='metric-card'><h3>Political</h3>"
            f"<p>{cls.political.title()}</p></div>",
            unsafe_allow_html=True,
        )
        
    badges = "".join(f"<span class='topic-badge'>{t.title()}</span>" for t in cls.topics) or "<p>None</p>"
    with r2[1]:
        st.markdown(
            f"<div class='metric-card'><h3>Topics</h3>{badges}</div>",
            unsafe_allow_html=True,
        )


st.header("Summarization")
if not run_sum:
    st.info("Summarization is turned off.")
else:
    summary = sum_fut.result()
    st.write(summary or "No summary could be generated.")
