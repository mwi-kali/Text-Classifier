from __future__ import annotations

import argparse
import sys
from pathlib import Path

from text_classifier import config
from text_classifier.ingestion import IngestionError, extract
from text_classifier.schemas import AnalysisRequest
from text_classifier.service import analyze
from text_classifier.summarization import Summarizer
from text_classifier.utils import configure_logging, is_url

DEFAULT_CORPUS = Path("eval/corpus.jsonl")


def _summary_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--chunk-tokens", type=int, default=config.SUMMARY_CHUNK_TOKENS)
    p.add_argument("--overlap-tokens", type=int, default=config.SUMMARY_OVERLAP_TOKENS)
    p.add_argument("--batch-size", type=int, default=config.SUMMARY_BATCH_SIZE)


def _evaluate(args) -> int:
    from text_classifier.evaluation import evaluate, load_corpus, to_markdown

    summ = Summarizer(
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        batch_size=args.batch_size,
    )
    rows = evaluate(summ, load_corpus(args.corpus))
    table = to_markdown(rows)
    out = Path(args.report_dir) / "summary_eval.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"# Summarisation evaluation\n\nModel `{config.SUMMARY_MODEL}`, chunk {args.chunk_tokens} tokens, "
        f"overlap {args.overlap_tokens}, batch {args.batch_size}.\n\n{table}\n"
    )
    print(out.read_text())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="text-classifier")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="classify and/or summarise a file, URL or text")
    src = a.add_mutually_exclusive_group(required=True)
    src.add_argument("source", nargs="?", help="path to a PDF/TXT/MD/HTML file, or a URL")
    src.add_argument("--text")
    a.add_argument("--no-classify", action="store_true")
    a.add_argument("--no-summary", action="store_true")
    a.add_argument("--map-reduce", action="store_true", help="map-reduce summarisation")
    a.add_argument("--render-js", action="store_true", help="render the URL in headless Chromium")
    a.add_argument("--no-tables", action="store_true")
    a.add_argument("--no-lists", action="store_true")
    a.add_argument("--no-code", action="store_true")
    _summary_args(a)

    e = sub.add_parser("evaluate", help="summary quality + performance on a small corpus")
    e.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    e.add_argument("--report-dir", type=Path, default=Path("reports"))
    _summary_args(e)

    s = sub.add_parser("serve", help="run the FastAPI service")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    configure_logging()

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("text_classifier.api:app", host=args.host, port=args.port)
        return 0
    if args.cmd == "evaluate":
        return _evaluate(args)

    options = {
        "classify": not args.no_classify,
        "summarize": not args.no_summary,
        "summary_mode": "map-reduce" if args.map_reduce else "abstractive",
        "include_tables": not args.no_tables,
        "include_lists": not args.no_lists,
        "include_code": not args.no_code,
        "render_js": True if args.render_js else None,
    }
    custom = (args.chunk_tokens, args.overlap_tokens, args.batch_size) != (
        config.SUMMARY_CHUNK_TOKENS,
        config.SUMMARY_OVERLAP_TOKENS,
        config.SUMMARY_BATCH_SIZE,
    )
    summarizer = (
        Summarizer(
            chunk_tokens=args.chunk_tokens,
            overlap_tokens=args.overlap_tokens,
            batch_size=args.batch_size,
        )
        if custom
        else None
    )
    try:
        if args.text:
            req = AnalysisRequest(text=args.text, **options)
        elif is_url(args.source):
            req = AnalysisRequest(url=args.source, **options)
        else:
            req = AnalysisRequest(text=extract(args.source, is_url_source=False).text, **options)
        print(analyze(req, summarizer=summarizer).model_dump_json(indent=2))
    except (IngestionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
