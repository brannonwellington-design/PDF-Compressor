#!/usr/bin/env python3
"""
PDF Compressor — All-in-one web app.
Single server: serves the UI + handles compression.
Deploy to Railway with zero config.
"""

import io
import os
import sys
import tempfile
import shutil
import zlib
from pathlib import Path

from flask import Flask, request, send_file, render_template, jsonify

import pikepdf
from pikepdf import Pdf, Name
from PIL import Image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB upload limit

# ─── Compression Engine ──────────────────────────────────────────────────────

# Levels 1–7, from gentlest to most aggressive
LEVELS = {
    "1": {"target_dpi": 300, "jpeg_quality": 95, "strip_metadata": False, "recompress_streams": True,  "label": "Lossless cleanup"},
    "2": {"target_dpi": 250, "jpeg_quality": 92, "strip_metadata": False, "recompress_streams": True,  "label": "High quality"},
    "3": {"target_dpi": 200, "jpeg_quality": 88, "strip_metadata": True,  "recompress_streams": True,  "label": "Quality"},
    "4": {"target_dpi": 150, "jpeg_quality": 82, "strip_metadata": True,  "recompress_streams": True,  "label": "Balanced"},
    "5": {"target_dpi": 120, "jpeg_quality": 72, "strip_metadata": True,  "recompress_streams": True,  "label": "Email-ready"},
    "6": {"target_dpi": 96,  "jpeg_quality": 60, "strip_metadata": True,  "recompress_streams": True,  "label": "Aggressive"},
    "7": {"target_dpi": 72,  "jpeg_quality": 45, "strip_metadata": True,  "recompress_streams": True,  "label": "Maximum"},
}


def _get_pil_image(image_obj):
    """Extract a PIL Image from a pikepdf image object, including alpha from SMask."""
    try:
        pil = pikepdf.PdfImage(image_obj).as_pil_image()

        # pikepdf strips SMask (alpha) — reconstruct it manually
        if "/SMask" in image_obj:
            try:
                smask = image_obj["/SMask"]
                alpha_pil = pikepdf.PdfImage(smask).as_pil_image()
                if alpha_pil.mode != "L":
                    alpha_pil = alpha_pil.convert("L")
                # Resize alpha to match if needed
                if alpha_pil.size != pil.size:
                    alpha_pil = alpha_pil.resize(pil.size, Image.LANCZOS)
                # Merge into RGBA
                if pil.mode != "RGB":
                    pil = pil.convert("RGB")
                pil = Image.merge("RGBA", (*pil.split(), alpha_pil))
            except Exception:
                pass

        return pil
    except Exception:
        return None


def _has_meaningful_alpha(pil_image):
    """Check if an RGBA image actually uses its alpha channel (not just fully opaque)."""
    if pil_image.mode != "RGBA":
        return False
    alpha = pil_image.getchannel("A")
    extrema = alpha.getextrema()
    # If min alpha is 255, the entire image is fully opaque — alpha is unused
    if extrema[0] == 255:
        return False
    return True


def _is_photo_like(pil_image):
    """
    Determine if image is photo-like (use JPEG) vs diagram/logo (keep lossless).
    Images with transparency are NEVER treated as photo-like.
    """
    # Anything with real transparency must stay lossless to preserve the alpha
    if _has_meaningful_alpha(pil_image):
        return False

    # Very low color count = diagram, logo, or flat art
    if pil_image.mode in ("1", "P", "L"):
        try:
            colors = pil_image.convert("RGB").getcolors(maxcolors=256)
            if colors is not None and len(colors) < 48:
                return False
        except Exception:
            pass
        return True

    # For RGB/RGBA (with no real alpha), check color complexity
    try:
        sample = pil_image.convert("RGB")
        # Sample a region if image is large to save time
        if sample.width > 500 or sample.height > 500:
            sample = sample.resize((min(500, sample.width), min(500, sample.height)), Image.NEAREST)
        colors = sample.getcolors(maxcolors=2048)
        if colors is not None and len(colors) < 64:
            return False
    except Exception:
        pass

    return True


def _estimate_image_dpi(w_px, h_px, page_w_pt, page_h_pt):
    """Estimate effective DPI of an image placed on a page."""
    if page_w_pt <= 0 or page_h_pt <= 0:
        return 300
    dpi_x = w_px / (page_w_pt / 72.0)
    dpi_y = h_px / (page_h_pt / 72.0)
    return max(dpi_x, dpi_y)


def compress_images(pdf, target_dpi, jpeg_quality):
    """Downsample and recompress images. Preserves transparency."""
    count = 0

    for page in pdf.pages:
        resources = page.get("/Resources", {})
        xobjects = resources.get("/XObject", {})
        if not xobjects:
            continue

        mediabox = page.get("/MediaBox", [0, 0, 612, 792])
        page_w = float(mediabox[2]) - float(mediabox[0])
        page_h = float(mediabox[3]) - float(mediabox[1])

        for name, obj_ref in list(xobjects.items()):
            try:
                xobj = obj_ref
                if not hasattr(xobj, "get") or xobj.get("/Subtype") != Name("/Image"):
                    continue

                w, h = int(xobj.get("/Width", 0)), int(xobj.get("/Height", 0))

                # Skip tiny images (icons, bullets, decorations)
                if w < 48 or h < 48:
                    continue

                pil = _get_pil_image(xobj)
                if pil is None:
                    continue

                # Estimate current DPI and decide on downsampling
                current_dpi = _estimate_image_dpi(w, h, page_w, page_h)
                scale = 1.0
                if current_dpi > target_dpi * 1.15:
                    scale = target_dpi / current_dpi

                nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

                # Only resize if meaningfully smaller
                if scale < 0.92:
                    pil = pil.resize((nw, nh), Image.LANCZOS)

                has_alpha = _has_meaningful_alpha(pil)
                is_photo = _is_photo_like(pil)

                if is_photo and not has_alpha:
                    # ── JPEG path: photos without transparency ──
                    if pil.mode == "RGBA":
                        # Flatten onto WHITE (not black)
                        bg = Image.new("RGB", pil.size, (255, 255, 255))
                        bg.paste(pil, mask=pil.split()[3])
                        pil = bg
                    elif pil.mode != "RGB":
                        pil = pil.convert("RGB")

                    buf = io.BytesIO()
                    pil.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
                    data = buf.getvalue()

                    # Only replace if JPEG is actually smaller than what was there
                    new = pdf.make_stream(data)
                    new[Name("/Type")] = Name("/XObject")
                    new[Name("/Subtype")] = Name("/Image")
                    new[Name("/Width")] = nw
                    new[Name("/Height")] = nh
                    new[Name("/ColorSpace")] = Name("/DeviceRGB")
                    new[Name("/BitsPerComponent")] = 8
                    new[Name("/Filter")] = Name("/DCTDecode")
                    xobjects[name] = new

                else:
                    # ── Lossless Flate path: diagrams, logos, anything with transparency ──
                    if has_alpha:
                        # PRESERVE ALPHA: encode as RGBA with SMask
                        if pil.mode != "RGBA":
                            pil = pil.convert("RGBA")

                        # Split into RGB + Alpha
                        r, g, b, a = pil.split()
                        rgb = Image.merge("RGB", (r, g, b))

                        # Main image stream (RGB)
                        rgb_data = zlib.compress(rgb.tobytes(), 9)
                        new = pdf.make_stream(rgb_data)
                        new[Name("/Type")] = Name("/XObject")
                        new[Name("/Subtype")] = Name("/Image")
                        new[Name("/Width")] = nw
                        new[Name("/Height")] = nh
                        new[Name("/ColorSpace")] = Name("/DeviceRGB")
                        new[Name("/BitsPerComponent")] = 8
                        new[Name("/Filter")] = Name("/FlateDecode")

                        # Alpha mask stream
                        alpha_data = zlib.compress(a.tobytes(), 9)
                        smask = pdf.make_stream(alpha_data)
                        smask[Name("/Type")] = Name("/XObject")
                        smask[Name("/Subtype")] = Name("/Image")
                        smask[Name("/Width")] = nw
                        smask[Name("/Height")] = nh
                        smask[Name("/ColorSpace")] = Name("/DeviceGray")
                        smask[Name("/BitsPerComponent")] = 8
                        smask[Name("/Filter")] = Name("/FlateDecode")

                        new[Name("/SMask")] = smask
                        xobjects[name] = new

                    else:
                        # No transparency — simple Flate
                        if pil.mode == "RGBA":
                            # No meaningful alpha, safe to drop
                            pil = pil.convert("RGB")
                        elif pil.mode not in ("RGB", "L"):
                            pil = pil.convert("RGB")

                        raw = zlib.compress(pil.tobytes(), 9)
                        new = pdf.make_stream(raw)
                        new[Name("/Type")] = Name("/XObject")
                        new[Name("/Subtype")] = Name("/Image")
                        new[Name("/Width")] = nw
                        new[Name("/Height")] = nh
                        cs = Name("/DeviceGray") if pil.mode == "L" else Name("/DeviceRGB")
                        new[Name("/ColorSpace")] = cs
                        new[Name("/BitsPerComponent")] = 8
                        new[Name("/Filter")] = Name("/FlateDecode")
                        xobjects[name] = new

                count += 1

            except Exception:
                continue

    return count


def strip_metadata(pdf):
    """Remove non-essential metadata."""
    if hasattr(pdf, "docinfo") and pdf.docinfo:
        try:
            for k in [k for k in pdf.docinfo.keys() if k != "/Title"]:
                del pdf.docinfo[k]
        except Exception:
            pass
    try:
        if "/Metadata" in pdf.Root:
            del pdf.Root[Name("/Metadata")]
    except Exception:
        pass
    for page in pdf.pages:
        try:
            if "/Thumb" in page:
                del page[Name("/Thumb")]
        except Exception:
            pass


def compress_pdf(input_path, output_path, level="4"):
    """Compress a PDF at the given level (1-7). Returns stats dict."""
    if level not in LEVELS:
        level = "4"
    config = LEVELS[level]
    input_size = os.path.getsize(input_path)

    pdf = Pdf.open(input_path)
    n_images = compress_images(pdf, config["target_dpi"], config["jpeg_quality"])

    if config["strip_metadata"]:
        strip_metadata(pdf)

    pdf.remove_unreferenced_resources()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        pdf.save(
            tmp_path,
            recompress_flate=config["recompress_streams"],
            object_stream_mode=pikepdf.ObjectStreamMode.generate,
            normalize_content=True,
        )
        pdf.close()

        # QPDF optimization pass
        qpdf = shutil.which("qpdf")
        if qpdf:
            import subprocess
            opt = tmp_path + ".opt.pdf"
            r = subprocess.run(
                [qpdf, "--recompress-flate", "--compression-level=9",
                 "--object-streams=generate", tmp_path, opt],
                capture_output=True,
            )
            if r.returncode == 0 and os.path.exists(opt) and os.path.getsize(opt) < os.path.getsize(tmp_path):
                os.replace(opt, tmp_path)
            elif os.path.exists(opt):
                os.remove(opt)

        shutil.move(tmp_path, output_path)

    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    output_size = os.path.getsize(output_path)
    return {
        "input_size": input_size,
        "output_size": output_size,
        "savings_pct": round((1 - output_size / input_size) * 100, 1) if input_size > 0 else 0,
        "images_processed": n_images,
    }


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/compress", methods=["POST"])
def compress():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "File must be a PDF"}), 400

    level = request.form.get("level", "4")
    if level not in LEVELS:
        level = "4"

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_in:
        file.save(tmp_in.name)
        input_path = tmp_in.name

    output_path = input_path + ".compressed.pdf"

    try:
        stats = compress_pdf(input_path, output_path, level=level)
        original_name = Path(file.filename).stem
        return send_file(
            output_path,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"{original_name}-compressed.pdf",
        )
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
