# Sample Data

This directory holds sample images and audio files for the demo.

## Quick Setup

Run the download script:

```bash
python download_samples.py
```

This fetches free-to-use images from Unsplash and audio from Pixabay.

## Manual Setup

If the download script fails (URLs can change), grab your own samples:

1. **cat_relaxing.jpg** — any photo of a cat (JPEG, ~640px)
2. **city_skyline.jpg** — any city/skyline photo (JPEG, ~640px)
3. **acoustic_guitar.mp3** — short guitar clip (MP3, ~10-30s)
4. **bird_chirping.mp3** — bird sounds clip (MP3, ~10-30s)

Place them in this `sample_data/` directory.

## Why These Files?

They demonstrate cross-modal similarity:

- Text "a cat sleeping in the sun" should score high similarity with `cat_relaxing.jpg`
- Text "acoustic guitar melody" should score high with `acoustic_guitar.mp3`
- `cat_relaxing.jpg` and `city_skyline.jpg` should score lower similarity (different subjects)

This is the key insight: Gemini Embedding 2 maps all modalities into the same vector space,
so semantic meaning is preserved even across different media types.
