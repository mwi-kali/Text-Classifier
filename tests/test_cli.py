import text_classifier.service as service
from text_classifier.classification import Classifier
from text_classifier.cli import main
from text_classifier.summarization import Summarizer


def test_cli_text(monkeypatch, fakes, article):
    sent, zs, summ = fakes
    monkeypatch.setattr(service, "Classifier", lambda: Classifier(sent, zs))
    monkeypatch.setattr(service, "Summarizer", lambda: Summarizer(summ, 30))
    assert main(["analyze", "--text", article]) == 0


def test_cli_file_error():
    assert main(["analyze", "/no/such.txt"]) == 1


def test_cli_file(tmp_path, article, monkeypatch, fakes):
    sent, zs, summ = fakes
    monkeypatch.setattr(service, "Classifier", lambda: Classifier(sent, zs))
    monkeypatch.setattr(service, "Summarizer", lambda: Summarizer(summ, 30))
    p = tmp_path / "a.md"
    p.write_text(article)
    assert main(["analyze", str(p), "--no-classify"]) == 0
