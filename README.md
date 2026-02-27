# PDF Compressor

Smart PDF compression that preserves live text, hyperlinks, and image clarity.

## Deploy to Railway (5 minutes)

1. **Create a GitHub repo**
   - Go to github.com → New Repository → name it `pdf-compressor`
   - Upload all the files in this folder to that repo

2. **Deploy on Railway**
   - Go to [railway.app](https://railway.app) and sign in with GitHub
   - Click **New Project** → **Deploy from GitHub Repo**
   - Select your `pdf-compressor` repo
   - Railway will automatically detect the Dockerfile and start building
   - Wait ~2 minutes for the build to finish

3. **Get your public URL**
   - In Railway, go to your project → **Settings** → **Networking**
   - Click **Generate Domain** — you'll get a URL like `pdf-compressor-production.up.railway.app`
   - Share this URL with your coworkers. Done.

## What it does

- **Image recompression**: Downsamples high-DPI images and recompresses as optimized JPEG or Flate
- **Stream optimization**: Recompresses internal PDF streams
- **Metadata stripping**: Removes non-essential metadata and thumbnails
- **Object deduplication**: Removes unreferenced and duplicate objects
- **QPDF pass**: Runs a final optimization pass with qpdf

## Three presets

| Preset | Image DPI | JPEG Quality | Best for |
|--------|-----------|-------------|----------|
| Email-Ready | 120 | 72% | Getting under 5 MB for email |
| Balanced | 150 | 82% | Everyday use |
| High Quality | 200 | 90% | When you need the images to look great |

## Run locally

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:8080
```
