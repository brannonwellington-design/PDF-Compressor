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

PRESETS = {
    "email": {"target_dpi": 120, "jpeg_quality": 72, "strip_metadata": True, "recompress_streams": True},
    "balanced": {"target_dpi": 150, "jpeg_quality": 82, "strip_metadata": True, "recompress_streams": True},
    "quality": {"target_dpi": 200, "jpeg_quality": 90, "strip_metadata": False, "recompress_streams": True},
}


def _get_pil_image(image_obj):
    try:
        return pikepdf.PdfImage(image_obj).as_pil_image()
    except Exception:
        return None


def _is_photo_like(pil_image):
    if pil_image.mode in ("1", "P"):
        colors = pil_image.convert("RGB").getcolors(maxcolors=256)
        if colors is not None and len(colors) < 32:
            return False
    if pil_image.mode == "RGBA":
        if pil_image.getchannel("A").getextrema() != (255, 255):
            return False
    if pil_image.mode in ("RGB", "RGBA"):
        try:
            colors = pil_image.convert("RGB").getcolors(maxcolors=1024)
            if colors is not None and len(colors) < 64:
                return False
        except Exception:
            pass
    return True


def compress_images(pdf, target_dpi, jpeg_quality):
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
                if w < 32 or h < 32:
                    continue

                pil = _get_pil_image(xobj)
                if pil is None:
                    continue

                dpi = max(w / (page_w / 72.0), h / (page_h / 72.0))
                scale = min(1.0, target_dpi / dpi) if dpi > target_dpi * 1.2 else 1.0
                nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

                if scale < 0.95:
                    pil = pil.resize((nw, nh), Image.LANCZOS)

                if _is_photo_like(pil):
                    if pil.mode == "RGBA":
                        bg = Image.new("RGB", pil.size, (255, 255, 255))
                        bg.paste(pil, mask=pil.split()[3])
                        pil = bg
                    elif pil.mode != "RGB":
                        pil = pil.convert("RGB")
                    buf = io.BytesIO()
                    pil.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
                    new = pdf.make_stream(buf.getvalue())
                    new[Name("/Type")] = Name("/XObject")
                    new[Name("/Subtype")] = Name("/Image")
                    new[Name("/Width")], new[Name("/Height")] = nw, nh
                    new[Name("/ColorSpace")] = Name("/DeviceRGB")
                    new[Name("/BitsPerComponent")] = 8
                    new[Name("/Filter")] = Name("/DCTDecode")
                    xobjects[name] = new
                else:
                    if pil.mode not in ("RGB", "L"):
                        pil = pil.convert("RGB")
                    compressed = zlib.compress(pil.tobytes(), 9)
                    new = pdf.make_stream(compressed)
                    new[Name("/Type")] = Name("/XObject")
                    new[Name("/Subtype")] = Name("/Image")
                    new[Name("/Width")], new[Name("/Height")] = nw, nh
                    new[Name("/ColorSpace")] = Name("/DeviceGray") if pil.mode == "L" else Name("/DeviceRGB")
                    new[Name("/BitsPerComponent")] = 8
                    new[Name("/Filter")] = Name("/FlateDecode")
                    xobjects[name] = new

                count += 1
            except Exception:
                continue
    return count


def strip_metadata(pdf):
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


def compress_pdf(input_path, output_path, preset="balanced"):
    config = PRESETS[preset]
    input_size = os.path.getsize(input_path)

    pdf = Pdf.open(input_path)
    n_images = compress_images(pdf, config["target_dpi"], config["jpeg_quality"])

    if config["strip_metadata"]:
        strip_metadata(pdf)

    pdf.remove_unreferenced_resources()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        pdf.save(tmp_path, recompress_flate=config["recompress_streams"],
                 object_stream_mode=pikepdf.ObjectStreamMode.generate, normalize_content=True)
        pdf.close()

        qpdf = shutil.which("qpdf")
        if qpdf:
            import subprocess
            opt = tmp_path + ".opt.pdf"
            r = subprocess.run([qpdf, "--recompress-flate", "--compression-level=9",
                                "--object-streams=generate", tmp_path, opt], capture_output=True)
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

    preset = request.form.get("preset", "balanced")
    if preset not in PRESETS:
        preset = "balanced"

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_in:
        file.save(tmp_in.name)
        input_path = tmp_in.name

    output_path = input_path + ".compressed.pdf"

    try:
        stats = compress_pdf(input_path, output_path, preset=preset)
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
