# Personal Thought Archive

Personal Thought Archive turns an exported X/Twitter archive into a private,
searchable collection of thoughts, ideas, and writing patterns.

It began as a data-science learning project and grew into a small AI application
with data cleaning, exploratory analysis, semantic search, retrieval-augmented
generation (RAG), and a writing assistant.

## What it can do

- Parse X/Twitter's JavaScript-wrapped tweet archive.
- Clean, deduplicate, and validate tweet records.
- Explore posting patterns in a Jupyter notebook.
- Search tweets using exact keywords.
- Search by meaning using API-generated embeddings.
- Answer questions using retrieved tweets as cited evidence.
- Generate three writing drafts using relevant original posts as style examples.

## Project flow

```text
tweets.js
    ↓
Parse and clean
    ↓
tweets_clean.json
    ↓
Create embeddings
    ↓
Semantic retrieval
    ├── Search similar tweets
    ├── Ask My Archive (RAG)
    └── Writing Assistant
```

## Project structure

```text
personal-thought-archive/
├── data/
│   ├── raw/              # Private archive files (ignored by Git)
│   └── processed/        # Generated datasets and vectors (ignored by Git)
├── notebooks/
│   └── 01_archive_exploration.ipynb
├── src/
│   ├── parse_tweets.py
│   ├── validate_tweets.py
│   ├── search_tweets.py
│   ├── build_api_embeddings.py
│   ├── api_semantic_search.py
│   ├── ask_archive.py
│   └── writing_assistant.py
└── tests/
```

## Setup

This project uses Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy only the tweet archive file into:

```text
data/raw/tweets.js
```

Do not commit the original archive. It may contain personal information.

## 1. Parse and validate the archive

```bash
python src/parse_tweets.py
python src/validate_tweets.py
```

The cleaned dataset is written to `data/processed/tweets_clean.json`.

## 2. Explore and search

Open `notebooks/01_archive_exploration.ipynb` in JupyterLab for exploratory
analysis, or use keyword search from the terminal:

```bash
python src/search_tweets.py "artificial intelligence"
```

## 3. Build the semantic-search index

Set an OpenAI API key in the current terminal without saving it in the project:

```bash
read -s OPENAI_API_KEY
export OPENAI_API_KEY
```

Then build the embedding index:

```bash
python src/build_api_embeddings.py
```

Search the archive by meaning:

```bash
python src/api_semantic_search.py "learning through projects" --type original
```

## 4. Ask My Archive

```bash
python src/ask_archive.py "What have I written about AI tools?"
```

The script retrieves relevant authored tweets and asks a language model to
produce an evidence-grounded answer with numbered tweet citations.

## 5. Generate writing drafts

```bash
python src/writing_assistant.py \
  "Many AI products solve convenience problems instead of meaningful problems"
```

The writing assistant retrieves relevant original posts as style examples and
returns three new drafts. Generated drafts should always be reviewed and edited
before posting.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests use synthetic records and do not upload the private archive.

## Privacy

The following are intentionally excluded from Git:

- Raw X/Twitter archive data
- Cleaned tweet datasets
- Generated embedding indexes
- API keys and `.env` files
- Virtual environments and downloaded model files

API-backed features send the query and selected tweet examples to the configured
API. Review the provider's privacy and retention settings before using personal
data.

## What I learned

- Schema inspection, parsing, cleaning, and validation
- Exploratory data analysis with pandas and Matplotlib
- Embeddings, cosine similarity, and semantic search
- Retrieval-augmented generation and source grounding
- Prompt design, evaluation, and privacy-aware development

## Possible next steps

- Add a small user interface.
- Improve retrieval quality and evaluation.
- Detect forgotten ideas and recurring themes.
- Track how opinions change over time.
