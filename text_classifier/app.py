from __future__ import annotations

import logging

import streamlit as st

from text_classifier import config
from text_classifier.classification import Classifier
from text_classifier.ingestion import IngestionError, bytes_to_text
from text_classifier.models import sentiment, summarizer, zero_shot
from text_classifier.schemas import AnalysisRequest
from text_classifier.service import analyze
from text_classifier.summarization import Summarizer
from text_classifier.utils import configure_logging, is_url

st.set_page_config(page_title="Text Classifier & Summarizer", page_icon="📝", layout="wide")
configure_logging()

st.markdown(
    """
    <style>
      .block-container { padding: 1rem 2rem; max-width: 1400px; }
      .metric-card { background: rgba(127,127,127,.08); padding: 1rem; border-radius: 10px;
                     margin-bottom: 1rem; text-align: center; }
      .metric-card h3 { margin: 0; opacity: .7; font-size: .8rem; text-transform: uppercase;
                        letter-spacing: .06em; }
      .metric-card p { margin: .25rem 0 0; font-size: 1.4rem; font-weight: 700; }
      .metric-card small { opacity: .6; }
      .topic-badge { display: inline-block; background: rgba(127,127,127,.18); padding: .2rem .55rem;
                     border-radius: 999px; margin: 2px 4px; font-size: .85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading models (first run downloads them)…")
def load_engines() -> tuple[Classifier, Summarizer]:
    """Load each model once per server process instead of once per click."""
    return Classifier(sentiment(), zero_shot()), Summarizer(summarizer())


def card(title: str, value: str, score: float | None = None, color: str | None = None) -> str:
    style = f" style='color:{color}'" if color else ""
    conf = f"<small>{score:.0%} confidence</small>" if score is not None else ""
    return f"<div class='metric-card'><h3>{title}</h3><p{style}>{value}</p>{conf}</div>"


st.title("Text Classifier & Summarizer")
st.write(
    "Upload a PDF/TXT file, paste text, or enter a web/YouTube URL to classify and summarise it."
)

with st.sidebar:
    st.header("Ingestion")
    source_type = st.radio("Pick a source type", ["File", "URL", "Text"])
    uploaded = (
        st.file_uploader(
            "Upload a PDF, TXT, MD or HTML file", type=sorted(config.SUPPORTED_FILE_EXTENSIONS)
        )
        if source_type == "File"
        else None
    )
    url = st.text_input("Web or YouTube URL") if source_type == "URL" else ""
    pasted = st.text_area("Paste text", height=200) if source_type == "Text" else ""
    st.divider()
    st.header("Settings")
    run_cls = st.checkbox("Run classification", value=True)
    run_sum = st.checkbox("Run summarization", value=True)
    mode = st.radio(
        "Summarization mode",
        ["Abstractive", "Map-Reduce"],
        help="Map-Reduce condenses long documents into one short summary.",
    )
    with st.expander("Extraction options (URLs)"):
        render_js = st.checkbox(
            "Render JavaScript in a headless browser",
            help="For single-page apps. Needs the browser extra.",
        )
        include_tables = st.checkbox("Include tables", value=True)
        include_lists = st.checkbox("Include lists", value=True)
        include_code = st.checkbox("Include code blocks", value=True)
    go = st.button("Process", use_container_width=True, type="primary")

if not go:
    st.stop()

try:
    options = {
        "classify": run_cls,
        "summarize": run_sum,
        "summary_mode": "map-reduce" if mode == "Map-Reduce" else "abstractive",
        "render_js": True if render_js else None,
        "include_tables": include_tables,
        "include_lists": include_lists,
        "include_code": include_code,
    }
    if source_type == "File":
        if uploaded is None:
            st.error("Please upload a file. Btw... it is in the sidebar. 👈")
            st.stop()
        request = AnalysisRequest(text=bytes_to_text(uploaded.getvalue(), uploaded.name), **options)
    elif source_type == "URL":
        if not is_url(url):
            st.error(
                "Please enter a valid http(s) URL. Btw... it is in the sidebar. Press Enter to lock it in. 👈"
            )
            st.stop()
        request = AnalysisRequest(url=url, **options)
    else:
        if not pasted.strip():
            st.error("Please paste some text. Btw... it is in the sidebar. 👈")
            st.stop()
        request = AnalysisRequest(text=pasted, **options)

    classifier, summ = load_engines()
    with st.spinner("Reading, classifying and summarising…"):
        result = analyze(request, classifier, summ)
except IngestionError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # never show a raw traceback to the user
    logging.getLogger(__name__).exception("analysis failed")
    st.error("Something went wrong while processing that source. Please try again.")
    with st.expander("Technical details"):
        st.code(f"{exc.__class__.__name__}: {exc}")
    st.stop()

st.caption(
    f"{result.words:,} words · "
    + " · ".join(f"{k} {v / 1000:.1f}s" for k, v in result.timings_ms.items())
)

if result.source:
    src = result.source
    bits = [src.get("title") or "", src.get("source_type", "").upper()]
    if src.get("rendered_js"):
        bits.append("rendered with JavaScript")
    if src.get("tables"):
        bits.append(f"{src['tables']} table(s)")
    st.caption(" · ".join(b for b in bits if b))
    if src.get("headings"):
        with st.expander(f"Document outline ({len(src['headings'])} sections)"):
            st.markdown("\n".join(f"- {h}" for h in src["headings"]))

st.header("Classification")
if result.classification is None:
    st.info("Classification is turned off.")
else:
    c = result.classification
    s = c.sentiment.label
    color = "#ef4444" if "neg" in s else "#10b981" if "pos" in s else None
    r1, r2 = st.columns(3), st.columns(2)
    r1[0].markdown(card("Sentiment", s.title(), c.sentiment.score, color), unsafe_allow_html=True)
    r1[1].markdown(card("Language", c.language.upper()), unsafe_allow_html=True)
    r1[2].markdown(card("Style", c.style.label.title(), c.style.score), unsafe_allow_html=True)
    r2[0].markdown(
        card("Political leaning (experimental)", c.political.label.title(), c.political.score),
        unsafe_allow_html=True,
    )
    badges = (
        "".join(
            f"<span class='topic-badge'>{t.label.title()} · {t.score:.0%}</span>" for t in c.topics
        )
        or "None"
    )
    r2[1].markdown(
        f"<div class='metric-card'><h3>Topics</h3>{badges}</div>", unsafe_allow_html=True
    )

st.header("Summarization")
if result.summary is None:
    st.info("Summarization is turned off.")
else:
    st.write(result.summary or "No summary could be generated.")
    st.download_button("Download summary", result.summary, file_name="summary.txt")
