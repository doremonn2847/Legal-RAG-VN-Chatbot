"""Download the public Zalo AI 2021 dataset and arrange the application's data files."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlretrieve
from zipfile import ZipFile, is_zipfile


DATA_DIR = Path(__file__).resolve().parent / "data"
DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/hariwh0/zaloai2021-legal-text-retrieval"
ARCHIVE_NAME = "zaloai2021.zip"
FILES = {
    "legal_corpus.json": "corpus/legal_corpus.json",
    "legal_corpus_hashmap.csv": "corpus/legal_corpus_hashmap.csv",
    "legal_corpus_legend.csv": "corpus/legal_corpus_legend.csv",
    "legal_corpus_merged_u256.csv": "corpus/legal_corpus_merged_u256.csv",
    "legal_corpus_merged_u369.csv": "corpus/legal_corpus_merged_u369.csv",
    "legal_corpus_original.csv": "corpus/legal_corpus_original.csv",
    "legal_corpus_splitted.csv": "corpus/legal_corpus_splitted.csv",
    "public_test_question.json": "test/public_test_question.json",
    "public_test_sample_submission.json": "test/public_test_sample_submission.json",
    "train_qna.csv": "train/train_qna.csv",
    "train_question_answer.json": "train/train_question_answer.json",
    "stopwords.txt": "utils/stopwords.txt",
}


def prepare_dataset(data_dir: Path = DATA_DIR):
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / ARCHIVE_NAME
    if not is_zipfile(archive_path):
        print("Downloading Zalo AI 2021 Legal Text Retrieval from Kaggle...")
        partial_path = archive_path.with_suffix(".zip.part")
        urlretrieve(DATASET_URL, partial_path)
        if not is_zipfile(partial_path):
            raise ValueError("Kaggle did not return a ZIP archive; check network access.")
        partial_path.replace(archive_path)

    with ZipFile(archive_path) as archive:
        members = {Path(name).name: name for name in archive.namelist()}
        missing = set(FILES) - members.keys()
        if missing:
            raise ValueError(f"Dataset archive is missing files: {sorted(missing)}")
        for name, relative_path in FILES.items():
            destination = data_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(members[name]) as source, destination.open("wb") as target:
                # Reading each member verifies its ZIP CRC; paths are from FILES only.
                shutil.copyfileobj(source, target)
            print(f"Prepared {relative_path}")

    with (data_dir / FILES["legal_corpus.json"]).open(encoding="utf-8") as source:
        corpus = json.load(source)
    if not isinstance(corpus, list) or not all("law_id" in law and "articles" in law for law in corpus):
        raise ValueError("Unexpected legal corpus schema.")
    articles = [article for law in corpus for article in law["articles"]]
    nonempty_articles = sum(bool(article.get("text", "").strip()) for article in articles)
    with (data_dir / FILES["train_qna.csv"]).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"question", "relevant_articles"}.issubset(reader.fieldnames or []):
            raise ValueError("Training CSV is missing question or relevant_articles columns.")
        training_questions = sum(1 for _ in reader)
    with archive_path.open("rb") as source:
        archive_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    with (data_dir / FILES["public_test_question.json"]).open(encoding="utf-8") as source:
        public_test = json.load(source)
    manifest = {
        "source": DATASET_URL,
        "archive_sha256": archive_sha256,
        "laws": len(corpus),
        "articles": len(articles),
        "indexable_articles": nonempty_articles,
        "training_questions": training_questions,
        "public_test_questions": len(public_test["items"]),
        "files": {path: (data_dir / path).stat().st_size for path in FILES.values()},
    }
    (data_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Validated {len(corpus):,} laws, {nonempty_articles:,} indexable articles, "
          f"and {training_questions:,} training questions.")
    return manifest


if __name__ == "__main__":
    prepare_dataset()
