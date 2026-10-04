import csv
from pathlib import Path

from questions.bank import load_bank, bank_questions, direct_question, structural_questions
from data.folder import FolderDataset
from data.phreshphish import label_to_int, DevIds
from data.split import stratified_split
import pandas as pd

BANK = Path(__file__).resolve().parents[1] / "src" / "questions" / "bank_v1.yaml"


def test_bank_contents():
    bank = load_bank(BANK)
    assert bank["version"] == "bank_v1"
    qs = bank_questions(bank, "S1")
    assert len(qs) == 15 and "q_ocr_brand_mismatch" not in qs
    assert len(bank_questions(bank, "S2")) == 17
    for q in qs.values():
        assert q["type"] == "noul" and set(q) <= {"type", "instructions", "criteria"}
    d = direct_question(bank)
    assert list(d) == ["q_direct"] and "`registrable_domain`" in d["q_direct"]["instructions"]
    st = structural_questions(bank)
    assert st["q_direct_score"]["type"] == "score" and len(st["q_direct_score"]["criteria"]) == 5
    assert st["q_direct_choice"]["type"] == "choice" and set(st["q_direct_choice"]["criteria"]) == {"phishing", "benign"}
    # every referenced field exists in the state schema
    fields = {"page_url", "registrable_domain", "forms", "has_password", "sensitive", "action_host",
              "meta.title", "meta.meta", "visible_text", "scripts", "entropy", "tokens",
              "favicon_host", "hosts.external_hosts", "screenshot_text"}
    import re
    for q in {**bank["questions"], **bank["structural"]}.values():
        for ref in re.findall(r"`([^`]+)`", q["instructions"]):
            assert ref in fields, ref


def test_folder_dataset(tmp_path):
    (tmp_path / "a").mkdir(); (tmp_path / "a" / "index.html").write_text("<p>x</p>")
    (tmp_path / "a" / "screenshot.jpg").write_bytes(b"\xff\xd8")
    (tmp_path / "b").mkdir(); (tmp_path / "b" / "index.html").write_text("<p>y</p>")
    with open(tmp_path / "meta.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["_id", "url", "label", "language", "source"])
        w.writerow(["a", "http://a", 1, "en", "t"]); w.writerow(["b", "http://b", 0, "de", "t"])
    ds = FolderDataset(tmp_path)
    assert len(ds) == 2 and ds.counts() == {"n": 2, "phishing": 1, "benign": 1, "with_html": 2, "with_screenshot": 1}
    assert ds.by_id("a").html() == "<p>x</p>" and ds.by_id("b").screenshot_path is None


def test_phreshphish_helpers(tmp_path):
    assert label_to_int("phish") == 1 and label_to_int("benign") == 0 and label_to_int("x") is None
    p = tmp_path / "meta.csv"
    p.write_text("_id,url,label,language,source,target,sha256\npp_1,http://u,1,en,phreshphish,,abc\n")
    dev = DevIds(p)
    assert dev.is_dev({"sha256": "abc", "url": "zzz"}) and dev.is_dev({"sha256": "q", "url": "http://u"})
    assert not dev.is_dev({"sha256": "q", "url": "r"})


def test_stratified_split_is_deterministic():
    df = pd.DataFrame({"id": [str(i) for i in range(300)],
                       "language": ["en"] * 200 + ["de"] * 100,
                       "html_length": list(range(300))})
    tr1, te1 = stratified_split(df, seed=42)
    tr2, te2 = stratified_split(df, seed=42)
    assert list(te1["id"]) == list(te2["id"])
    assert abs(len(te1) / 300 - 0.30) < 0.02
    assert set(tr1["id"]).isdisjoint(te1["id"])
