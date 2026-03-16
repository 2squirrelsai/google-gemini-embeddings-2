"""
Gemini Embedding 2 Explorer
---
Interactive demo for Google's first multimodal embedding model.
Embed text, images, and audio into a unified vector space and compare similarity.
"""

import os
import base64
import mimetypes
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import faiss
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PORT = int(os.getenv("PORT", "8000"))
DEFAULT_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
MODEL_ID = "gemini-embedding-2-preview"
SAMPLE_DIR = Path(__file__).parent / "sample_data"
SUPPORTED_DIMS = (768, 1536, 3072)

# Per-dimension FAISS index + metadata, populated at startup
search_indexes: dict[int, dict] = {}

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. "
        "Get one at https://aistudio.google.com/apikey "
        "and add it to your .env file."
    )

# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------
client = genai.Client(api_key=GOOGLE_API_KEY)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Gemini Embedding 2 Explorer",
    description="Multimodal embedding demo with similarity heatmap",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/sample_data", StaticFiles(directory="sample_data"), name="sample_data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cosine_similarity_matrix(embeddings: list[list[float]]) -> list[list[float]]:
    """Compute pairwise cosine similarity for a list of embedding vectors."""
    arr = np.array(embeddings)
    # L2 normalize each row
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    normed = arr / norms
    sim = (normed @ normed.T).tolist()
    return sim


def detect_mime_type(filename: str) -> str:
    """Guess MIME type from filename extension."""
    mime, _ = mimetypes.guess_type(filename)
    if mime:
        return mime
    ext = Path(filename).suffix.lower()
    fallback = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".pdf": "application/pdf",
    }
    return fallback.get(ext, "application/octet-stream")


async def embed_item(
    item: dict, dimensions: int, task_type: Optional[str] = None
) -> dict:
    """
    Embed a single item. Returns dict with label, type, and embedding values.

    item format:
      {"type": "text", "value": "some text"}
      {"type": "file", "filename": "cat.jpg", "data": <bytes>}

    task_type: optional Gemini task type (e.g. "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY")
    """
    config = types.EmbedContentConfig(
        output_dimensionality=dimensions,
        task_type=task_type,
    )

    if item["type"] == "text":
        result = client.models.embed_content(
            model=MODEL_ID,
            contents=item["value"],
            config=config,
        )
        return {
            "label": item["value"][:60],
            "modality": "text",
            "values": result.embeddings[0].values,
        }

    elif item["type"] == "file":
        mime = detect_mime_type(item["filename"])
        part = types.Part.from_bytes(data=item["data"], mime_type=mime)
        result = client.models.embed_content(
            model=MODEL_ID,
            contents=[part],
            config=config,
        )

        # Determine modality from mime type
        if mime.startswith("image"):
            modality = "image"
        elif mime.startswith("audio"):
            modality = "audio"
        elif mime.startswith("video"):
            modality = "video"
        elif mime == "application/pdf":
            modality = "document"
        else:
            modality = "file"

        return {
            "label": item["filename"],
            "modality": modality,
            "values": result.embeddings[0].values,
        }

    raise ValueError(f"Unknown item type: {item['type']}")


# ---------------------------------------------------------------------------
# Search index (built at startup)
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def build_search_index():
    """Embed all sample_data files at each supported dimension and build FAISS indexes."""
    sample_files = [
        f for f in SAMPLE_DIR.iterdir()
        if not f.name.startswith(".") and f.suffix.lower() != ".md"
    ]
    if not sample_files:
        print("  No sample files found — skipping search index build.")
        return

    print(f"  Building search index for {len(sample_files)} files...")

    for dim in SUPPORTED_DIMS:
        vectors = []
        metadata = []  # parallel list: filename, modality per vector
        for sf in sample_files:
            try:
                item = {"type": "file", "filename": sf.name, "data": sf.read_bytes()}
                result = await embed_item(item, dim)
                vec = np.array(result["values"], dtype=np.float32)
                # L2 normalize for inner-product search (cosine similarity)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
                vectors.append(vec)
                metadata.append({
                    "filename": sf.name,
                    "modality": result["modality"],
                })
            except Exception as e:
                print(f"    Warning: failed to embed {sf.name} at {dim}d — {e}")

        if vectors:
            matrix = np.stack(vectors).astype(np.float32)
            index = faiss.IndexFlatIP(dim)
            index.add(matrix)
            search_indexes[dim] = {"index": index, "metadata": metadata}
            print(f"    {dim}d index: {index.ntotal} vectors")

    print("  Search index ready.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("static/index.html").read_text()


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "default_dimensions": DEFAULT_DIMENSIONS,
        "sample_files": [f.name for f in SAMPLE_DIR.iterdir() if not f.name.startswith(".")],
        "index_ready": len(search_indexes) == len(SUPPORTED_DIMS),
    }


@app.post("/api/embed")
async def embed(
    texts: Optional[str] = Form(None),
    dimensions: Optional[int] = Form(None),
    files: list[UploadFile] = File(default=[]),
    use_samples: Optional[str] = Form(None),
):
    """
    Embed a mix of text strings and files, return embeddings + similarity matrix.

    - texts: newline-separated text strings
    - files: uploaded image/audio files
    - use_samples: comma-separated sample filenames to include
    - dimensions: output dimensionality (768, 1536, or 3072)
    """
    dims = dimensions or DEFAULT_DIMENSIONS
    if dims not in (768, 1536, 3072):
        raise HTTPException(400, "dimensions must be 768, 1536, or 3072")

    items = []

    # Parse text inputs
    if texts:
        for line in texts.strip().splitlines():
            line = line.strip()
            if line:
                items.append({"type": "text", "value": line})

    # Parse uploaded files
    # Audio limit: ~80 seconds max. At typical MP3 bitrates, ~1.5 MB ≈ 80s.
    MAX_AUDIO_BYTES = 1_500_000
    for f in files:
        if f.filename and f.size and f.size > 0:
            data = await f.read()
            mime = detect_mime_type(f.filename)
            if mime.startswith("audio") and len(data) > MAX_AUDIO_BYTES:
                raise HTTPException(
                    400,
                    f"Audio file '{f.filename}' is too large ({len(data)//1024}KB). "
                    f"Gemini Embedding 2 supports audio up to ~80 seconds. "
                    f"Please use a shorter clip.",
                )
            items.append({"type": "file", "filename": f.filename, "data": data})

    # Parse sample file references
    if use_samples:
        for name in use_samples.split(","):
            name = name.strip()
            if not name:
                continue
            path = SAMPLE_DIR / name
            if not path.exists():
                raise HTTPException(404, f"Sample file not found: {name}")
            items.append({
                "type": "file",
                "filename": name,
                "data": path.read_bytes(),
            })

    if len(items) < 2:
        raise HTTPException(400, "Provide at least 2 items to compare")

    if len(items) > 10:
        raise HTTPException(400, "Maximum 10 items per request")

    # Embed all items
    try:
        results = []
        for item in items:
            result = await embed_item(item, dims)
            results.append(result)
    except Exception as e:
        raise HTTPException(500, f"Embedding failed: {str(e)}")

    # Compute similarity
    all_embeddings = [r["values"] for r in results]
    similarity = cosine_similarity_matrix(all_embeddings)

    # Build response (strip raw vectors for readability, include dimension count)
    labels = []
    modalities = []
    for r in results:
        labels.append(r["label"])
        modalities.append(r["modality"])

    return JSONResponse({
        "labels": labels,
        "modalities": modalities,
        "dimensions": dims,
        "count": len(results),
        "similarity": similarity,
        # Include first 5 values of each embedding for inspection
        "embedding_previews": [r["values"][:5] for r in results],
    })


@app.post("/api/search")
async def search(
    query_text: Optional[str] = Form(None),
    query_file: Optional[UploadFile] = File(None),
    dimensions: Optional[int] = Form(None),
    top_k: Optional[int] = Form(5),
):
    """
    Semantic search: embed a query and retrieve the most similar items from the
    pre-indexed sample_data corpus using FAISS.
    """
    dims = dimensions or DEFAULT_DIMENSIONS
    if dims not in SUPPORTED_DIMS:
        raise HTTPException(400, f"dimensions must be one of {SUPPORTED_DIMS}")

    if dims not in search_indexes:
        raise HTTPException(503, "Search index not ready yet — try again shortly.")

    # Build query item (file takes priority)
    if query_file and query_file.filename and query_file.size and query_file.size > 0:
        data = await query_file.read()
        item = {"type": "file", "filename": query_file.filename, "data": data}
    elif query_text and query_text.strip():
        item = {"type": "text", "value": query_text.strip()}
    else:
        raise HTTPException(400, "Provide query_text or query_file")

    # Embed query
    try:
        result = await embed_item(item, dims, task_type="RETRIEVAL_QUERY")
    except Exception as e:
        raise HTTPException(500, f"Query embedding failed: {str(e)}")

    query_vec = np.array(result["values"], dtype=np.float32).reshape(1, -1)
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec /= norm

    idx_data = search_indexes[dims]
    k = min(top_k or 5, idx_data["index"].ntotal)
    scores, indices = idx_data["index"].search(query_vec, k)

    results_list = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx == -1:
            continue
        meta = idx_data["metadata"][idx]
        results_list.append({
            "rank": rank,
            "filename": meta["filename"],
            "modality": meta["modality"],
            "score": round(float(score), 6),
            "preview_url": f"/sample_data/{quote(meta['filename'])}",
        })

    return JSONResponse({
        "query_label": result["label"],
        "query_modality": result["modality"],
        "dimensions": dims,
        "results": results_list,
    })


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n  Gemini Embedding 2 Explorer")
    print(f"  Model: {MODEL_ID}")
    print(f"  Dimensions: {DEFAULT_DIMENSIONS}")
    print(f"  http://localhost:{PORT}\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
