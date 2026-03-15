#!/usr/bin/env python3
"""
Download sample data for the Gemini Embedding 2 Explorer demo.

Uses freely licensed images from Unsplash and audio from Pixabay.
Run this once before starting the app:

    python download_samples.py

All files are saved to sample_data/
"""

import urllib.request
import os
from pathlib import Path

SAMPLE_DIR = Path(__file__).parent / "sample_data"
SAMPLE_DIR.mkdir(exist_ok=True)

# --- Sample files ---
# Replace these URLs with your own preferred samples.
# The defaults below are from free/open-licensed sources.

SAMPLES = {
    # Images (Unsplash - free to use)
    "cat_relaxing.jpg": "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=640&q=80",
    "city_skyline.jpg": "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=640&q=80",

    # Audio (Pixabay - free to use, royalty-free)
    # NOTE: Pixabay URLs may change. If these fail, download manually from:
    #   https://pixabay.com/sound-effects/search/guitar/
    #   https://pixabay.com/sound-effects/search/bird/
    # Save as acoustic_guitar.mp3 and bird_chirping.mp3 in sample_data/
    "acoustic_guitar.mp3": "https://cdn.pixabay.com/audio/2022/01/20/audio_7a5710c49e.mp3",
    "bird_chirping.mp3": "https://cdn.pixabay.com/audio/2022/03/24/audio_1030a6a01b.mp3",
}


def download():
    for filename, url in SAMPLES.items():
        dest = SAMPLE_DIR / filename
        if dest.exists():
            print(f"  [skip] {filename} already exists")
            continue
        print(f"  [download] {filename} ...")
        try:
            urllib.request.urlretrieve(url, dest)
            size_kb = dest.stat().st_size / 1024
            print(f"  [done] {filename} ({size_kb:.0f} KB)")
        except Exception as e:
            print(f"  [error] {filename}: {e}")
            print(f"           Download manually and place in sample_data/{filename}")


if __name__ == "__main__":
    print("Downloading sample data...\n")
    download()
    print("\nDone! You can now run: python app.py")
