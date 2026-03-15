<div align="center">

# Gemini Embedding 2 Explorer

**Multimodal Semantic Search & Similarity — Text, Images, and Audio in One Vector Space**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Gemini Embedding 2](https://img.shields.io/badge/Gemini_Embedding_2-Public_Preview-F59E0B?logo=google&logoColor=white)](https://ai.google.dev/gemini-api/docs/embeddings)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-4285F4)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E.svg)](LICENSE)

<br />

*Embed text, images, and audio into a unified vector space. Search semantically across modalities or compare pairwise similarity — all in your browser.*

[Getting Started](#quick-start) &nbsp;&bull;&nbsp; [Features](#features) &nbsp;&bull;&nbsp; [How It Works](#how-it-works) &nbsp;&bull;&nbsp; [API Reference](#api-endpoints) &nbsp;&bull;&nbsp; [Resources](#resources)

</div>

---

## The Problem

Traditional search is keyword-based and single-modality. You can't easily ask *"find images that match this description"* or *"which audio clip sounds like this photo looks."* Multimodal embeddings change that — but the APIs are new and hard to explore without a hands-on tool.

## The Solution

Gemini Embedding 2 Explorer is an interactive demo that makes Google's first natively multimodal embedding model tangible. Upload images, paste text, drop audio files — and instantly see how the model understands semantic relationships across modalities.

---

## Features

### Semantic Search (FAISS-powered)

| Feature | Description |
|---------|-------------|
| **Text-to-Image Search** | Type "a cat sleeping in the sun" and retrieve the most similar images from the corpus |
| **Audio-to-Audio Search** | Upload an audio clip and find semantically similar sounds |
| **Cross-Modal Retrieval** | Search with any modality, retrieve results from any other modality |
| **Pre-Indexed Corpus** | Sample data is embedded and indexed at startup across all 3 dimension sizes |
| **Ranked Results** | Results displayed with similarity scores, media previews, and visual score bars |

### Pairwise Comparison

| Feature | Description |
|---------|-------------|
| **Interactive Heatmap** | Cosine similarity matrix with color-coded cells and hover tooltips |
| **Multi-Input** | Compare up to 10 items: text strings, uploaded files, and sample data |
| **Sorted Pair List** | All pairs ranked by similarity score |
| **Dimension Scaling** | Switch between 768, 1536, and 3072 dimensions (Matryoshka Representation Learning) |

### General

| Feature | Description |
|---------|-------------|
| **Zero Build Step** | Vanilla HTML/CSS/JS frontend — no Node.js, no bundler |
| **Sample Data Included** | 5 images + 3 audio clips ready to explore out of the box |
| **Dark Theme UI** | Clean, modern interface designed for focus |
| **Responsive** | Works on desktop and mobile browsers |

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/2squirrelsai/google-gemini-embeddings-2
cd gemini-embedding-2-explorer
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download sample data

```bash
python download_samples.py
```

This fetches sample images and audio clips from free sources. You can also add your own files to `sample_data/`.

### 4. Set your API key

Get a free key from [Google AI Studio](https://aistudio.google.com/apikey).

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Or create a `.env` file:

```
GOOGLE_API_KEY=your-api-key-here
```

### 5. Run

```bash
python app.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

> On first launch, the server builds FAISS search indexes by embedding all sample files at 3 dimension sizes (~24 API calls). You'll see **"Search index ready."** in the console when it's done.

---

## How It Works

```
┌──────────────────┐     ┌─────────────────────────┐     ┌──────────────────┐
│   Web UI         │────▶│   FastAPI Backend        │────▶│   Gemini API     │
│   (vanilla JS)   │◀────│                          │◀────│   embed_content()│
│                  │     │   /api/embed   (compare)  │     │                  │
│   Compare mode   │     │   /api/search  (FAISS)    │     │   gemini-        │
│   Search mode    │     │   /api/health             │     │   embedding-2    │
└──────────────────┘     └─────────────────────────┘     └──────────────────┘
                                    │
                              ┌─────▼─────┐
                              │   FAISS    │
                              │   Index    │
                              │ (in-memory)│
                              └───────────┘
```

**Compare mode:**
1. UI sends text, images, or audio to `/api/embed`
2. Backend embeds each item via Gemini Embedding 2
3. Cosine similarity is computed pairwise
4. Results render as an interactive heatmap

**Search mode:**
1. At startup, all `sample_data/` files are embedded and added to FAISS `IndexFlatIP` indexes (one per dimension size)
2. User submits a query (text or file) to `/api/search`
3. Query is embedded, L2-normalized, and searched against the FAISS index
4. Top-k results are returned with similarity scores and media previews

---

## Sample Data

The `sample_data/` directory includes files for immediate exploration:

| File | Type | Description |
|------|------|-------------|
| `cat-in-chair.jpeg` | Image | Cat lounging in a chair |
| `cat-in-window.jpeg` | Image | Cat sitting by a window |
| `cardinals-male-female.jpeg` | Image | Two cardinal birds |
| `birds-iceskating.jpeg` | Image | Artistic bird illustration |
| `city-skyline.jpeg` | Image | Urban skyline |
| `acoustic_guitar.mp3` | Audio | Acoustic guitar clip |
| `bird_chirping.mp3` | Audio | Bird sounds in nature |
| `Boots Runs This Block.mp3` | Audio | Music track |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/embed` | POST | Embed items and return pairwise similarity matrix |
| `/api/search` | POST | Semantic search against the pre-indexed corpus |
| `/api/health` | GET | Health check, model info, and index readiness |

---

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `GOOGLE_API_KEY` | — | **Required.** Your Gemini API key |
| `PORT` | `8000` | Server port |
| `EMBEDDING_DIMENSIONS` | `768` | Default output dimensions (768, 1536, or 3072) |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **AI** | Google GenAI SDK, Gemini Embedding 2 (`gemini-embedding-2-preview`) |
| **Vector Search** | FAISS (`IndexFlatIP` with L2-normalized vectors for cosine similarity) |
| **Frontend** | Vanilla HTML/CSS/JS — no build step, no framework |
| **Math** | NumPy for similarity computation |

---

## Project Structure

```
gemini-embedding-2-explorer/
├── app.py                 # FastAPI server, embedding logic, FAISS search
├── download_samples.py    # Download sample images & audio
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
├── sample_data/           # Sample images and audio (indexed at startup)
│   ├── cat-in-chair.jpeg
│   ├── cat-in-window.jpeg
│   ├── cardinals-male-female.jpeg
│   ├── birds-iceskating.jpeg
│   ├── city-skyline.jpeg
│   ├── acoustic_guitar.mp3
│   ├── bird_chirping.mp3
│   └── Boots Runs This Block.mp3
├── static/
│   └── index.html         # Web UI (compare + search modes)
├── LICENSE
└── README.md
```

---

## Key Concepts Demonstrated

- **Unified embedding space** — Text query "a cat sleeping" returns high similarity with `cat-in-chair.jpeg` even though one is text and the other is an image
- **Cross-modal retrieval** — Audio of a guitar has higher similarity to "acoustic music" than to "city traffic"
- **Matryoshka dimensions** — Compare 768 vs 3072 dimensions. Smaller vectors are faster and cheaper with minimal quality loss
- **FAISS vector search** — Production-grade similarity search over pre-indexed embeddings

---

## Limitations

- Gemini Embedding 2 is in **Public Preview** — the model ID and behavior may change
- Images: up to 6 per request, PNG/JPEG only
- Audio: natively embedded (no transcription needed), up to ~80 seconds
- Video: supported by the model but not included in this demo
- The embedding spaces of `gemini-embedding-001` and `gemini-embedding-2-preview` are **incompatible** — don't mix them

---

## License

MIT — see [LICENSE](LICENSE).

---

## Resources

- [Gemini Embedding 2 Announcement](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/)
- [Embeddings API Docs](https://ai.google.dev/gemini-api/docs/embeddings)
- [Google GenAI Python SDK](https://github.com/googleapis/python-genai)
- [FAISS Library](https://github.com/facebookresearch/faiss)
- [Google AI Studio (get API key)](https://aistudio.google.com/apikey)

---

<div align="center">

**Explore how AI understands the relationship between what you see, hear, and read.**

Built by [Tony Turner](https://github.com/tonyturner) / [2 Squirrels AI](https://2squirrels.ai)

</div>
