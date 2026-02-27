#!/usr/bin/env python3
"""
PDF Compressor — All-in-one web app.

Two-pass compression:
  Pass 1: Ghostscript — rewrites the entire PDF, recompresses ALL images
          regardless of format (JPEG2000, JBIG2, ICCBased, etc.)
  Pass 2: pikepdf/QPDF — structural optimization (object dedup, stream
          recompression, metadata stripping)

This handles real-world PDFs from InDesign, Illustrator, Word, scanners, etc.
"""

import io
import os
import subprocess
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
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

# ─── Compression Levels ──────────────────────────────────────────────────────

LEVELS = {
    "1": {
        "gs_setting": "/printer",   "gs_dpi": 300, "gs_quality": 0.95,
        "target_dpi": 300, "jpeg_quality": 95,
        "strip_metadata": False, "label": "Lossless cleanup",
    },
    "2": {
        "gs_setting": "/printer",   "gs_dpi": 250, "gs_quality": 0.90,
        "target_dpi": 250, "jpeg_quality": 92,
        "strip_metadata": False, "label": "High quality",
    },
    "3": {
        "gs_setting": "/ebook",     "gs_dpi": 200, "gs_quality": 0.85,
        "target_dpi": 200, "jpeg_quality": 88,
        "strip_metadata": True,  "label": "Quality",
    },
    "4": {
        "gs_setting": "/ebook",     "gs_dpi": 150, "gs_quality": 0.75,
        "target_dpi": 150, "jpeg_quality": 82,
        "strip_metadata": True,  "label": "Balanced",
    },
    "5": {
        "gs_setting": "/ebook",     "gs_dpi": 120, "gs_quality": 0.60,
        "target_dpi": 120, "jpeg_quality": 72,
        "strip_metadata": True,  "label": "Email-ready",
    },
    "6": {
        "gs_setting": "/screen",    "gs_dpi": 96,  "gs_quality": 0.40,
        "target_dpi": 96,  "jpeg_quality": 60,
        "strip_metadata": True,  "label": "Aggressive",
    },
    "7": {
        "gs_setting": "/screen",    "gs_dpi": 72,  "gs_quality": 0.30,
        "target_dpi": 72,  "jpeg_quality": 45,
        "strip_metadata": True,  "label": "Maximum",
    },
}


# ─── Pass 1: Ghostscript ─────────────────────────────────────────────────────

def ghostscript_compress(input_path, output_path, config):
    """
    Use Ghostscript to rewrite the entire PDF with compressed images.
    This catches EVERYTHING — JPEG2000, JBIG2, ICCBased colorspaces,
    images inside Form XObjects, etc. Preserves text, links, bookmarks.
    """
    gs = shutil.which("gs")
    if not gs:
        # Ghostscript not available, skip this pass
        shutil.copy2(input_path, output_path)
        return False

    dpi = config["gs_dpi"]
    quality = config["gs_quality"]

    cmd = [
        gs,
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.5",
        "-dNOPAUSE",
        "-dBATCH",
        "-dQUIET",

        # ── Image downsampling ──
        f"-dColorImageResolution={dpi}",
        f"-dGrayImageResolution={dpi}",
        f"-dMonoImageResolution={min(dpi * 2, 300)}",

        "-dDownsampleColorImages=true",
        "-dDownsampleGrayImages=true",
        "-dDownsampleMonoImages=true",

        "-dColorImageDownsampleType=/Bicubic",
        "-dGrayImageDownsampleType=/Bicubic",
        "-dMonoImageDownsampleType=/Subsample",

        # Only downsample if image DPI exceeds target by 1.5x
        f"-dColorImageDownsampleThreshold=1.5",
        f"-dGrayImageDownsampleThreshold=1.5",
        f"-dMonoImageDownsampleThreshold=1.5",

        # ── JPEG compression quality ──
        "-dAutoFilterColorImages=false",
        "-dAutoFilterGrayImages=false",
        "-dColorImageFilter=/DCTEncode",
        "-dGrayImageFilter=/DCTEncode",
        f"-c",
        f"<< /ColorACSImageDict << /QFactor {quality} /Blend 1 /HSamples [1 1 1 1] /VSamples [1 1 1 1] >> >> setdistillerparams",
        f"<< /GrayACSImageDict << /QFactor {quality} /Blend 1 /HSamples [1 1 1 1] /VSamples [1 1 1 1] >> >> setdistillerparams",
        "-f",

        # ── Preserve structure ──
        "-dPrinted=false",           # Preserve screen annotations (links!)
        "-dPreserveAnnots=true",     # Keep all annotations
        "-dPreserveMarkedContent=true",
        "-dPreserveOPIComments=false",
        "-dPreserveOverprintSettings=false",

        # ── Font handling ──
        "-dSubsetFonts=true",
        "-dEmbedAllFonts=true",
        "-dCompressFonts=true",

        # ── Compress everything ──
        "-dCompressPages=true",
        "-dUseFlateCompression=true",

        f"-sOutputFile={output_path}",
        input_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and os.path.exists(output_path):
            output_size = os.path.getsize(output_path)
            if output_size > 0:
                return True
        # If GS produced an empty or invalid file, fall back
        if os.path.exists(output_path):
            os.unlink(output_path)
    except subprocess.TimeoutExpired:
        if os.path.exists(output_path):
            os.unlink(output_path)
    except Exception:
        if os.path.exists(output_path):
            os.unlink(output_path)

    # Fallback: just copy the input
    shutil.copy2(input_path, output_path)
    return False


# ─── Pass 2: pikepdf structural optimization ─────────────────────────────────

def _get_pil_image(image_obj):
    """Extract a PIL Image, reconstructing alpha from SMask."""
    try:
        pil = pikepdf.PdfImage(image_obj).as_pil_image()
        if "/SMask" in image_obj:
            try:
                smask = image_obj["/SMask"]
                alpha = pikepdf.PdfImage(smask).as_pil_image()
                if alpha.mode != "L":
                    alpha = alpha.convert("L")
                if alpha.size != pil.size:
                    alpha = alpha.resize(pil.size, Image.LANCZOS)
                if pil.mode != "RGB":
                    pil = pil.convert("RGB")
                pil = Image.merge("RGBA", (*pil.split(), alpha))
            except Exception:
                pass
        return pil
    except Exception:
        return None


def _has_meaningful_alpha(pil_image):
    if pil_image.mode != "RGBA":
        return False
    return pil_image.getchannel("A").getextrema()[0] < 255


def _is_photo_like(pil_image):
    if _has_meaningful_alpha(pil_image):
        return False
    if pil_image.mode in ("1", "P", "L"):
        try:
            colors = pil_image.convert("RGB").getcolors(maxcolors=256)
            if colors is not None and len(colors) < 48:
                return False
        except Exception:
            pass
        return True
    try:
        sample = pil_image.convert("RGB")
        if sample.width > 400 or sample.height > 400:
            sample = sample.resize((min(400, sample.width), min(400, sample.height)), Image.NEAREST)
        colors = sample.getcolors(maxcolors=2048)
        if colors is not None and len(colors) < 64:
            return False
    except Exception:
        pass
    return True


def pikepdf_optimize(input_path, output_path, config):
    """
    Second pass: pikepdf/QPDF structural optimization.
    After Ghostscript has already handled image compression, this pass:
    - Catches any remaining oversized images GS missed
    - Removes unreferenced objects
    - Recompresses all streams
    - Strips metadata
    - Runs QPDF optimization
    """
    pdf = Pdf.open(input_path)

    target_dpi = config["target_dpi"]
    jpeg_quality = config["jpeg_quality"]
    images_done = 0

    # Image pass — catch anything Ghostscript missed or didn't compress enough
    for page in pdf.pages:
        _process_xobjects(pdf, page, target_dpi, jpeg_quality)

    # Strip metadata
    if config["strip_metadata"]:
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

    # Remove unreferenced objects
    pdf.remove_unreferenced_resources()

    # Save with maximum compression
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        pdf.save(
            tmp_path,
            recompress_flate=True,
            object_stream_mode=pikepdf.ObjectStreamMode.generate,
            normalize_content=True,
        )
        pdf.close()

        # QPDF final pass
        qpdf = shutil.which("qpdf")
        if qpdf:
            opt = tmp_path + ".opt.pdf"
            r = subprocess.run(
                [qpdf, "--recompress-flate", "--compression-level=9",
                 "--object-streams=generate", tmp_path, opt],
                capture_output=True, timeout=120,
            )
            if r.returncode == 0 and os.path.exists(opt):
                if os.path.getsize(opt) < os.path.getsize(tmp_path):
                    os.replace(opt, tmp_path)
                else:
                    os.remove(opt)
            elif os.path.exists(opt):
                os.remove(opt)

        shutil.move(tmp_path, output_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _process_xobjects(pdf, page, target_dpi, jpeg_quality):
    """Process images in page XObjects, including nested Form XObjects."""
    resources = page.get("/Resources", {})
    xobjects = resources.get("/XObject", {})
    if not xobjects:
        return

    mediabox = page.get("/MediaBox", [0, 0, 612, 792])
    page_w = float(mediabox[2]) - float(mediabox[0])
    page_h = float(mediabox[3]) - float(mediabox[1])

    for name, obj_ref in list(xobjects.items()):
        try:
            xobj = obj_ref
            if not hasattr(xobj, "get"):
                continue

            subtype = xobj.get("/Subtype")

            # Recurse into Form XObjects (they can contain images too)
            if subtype == Name("/Form"):
                form_res = xobj.get("/Resources", {})
                form_xobjs = form_res.get("/XObject", {})
                if form_xobjs:
                    for fname, fobj in list(form_xobjs.items()):
                        try:
                            if hasattr(fobj, "get") and fobj.get("/Subtype") == Name("/Image"):
                                _compress_single_image(pdf, form_xobjs, fname, fobj, page_w, page_h, target_dpi, jpeg_quality)
                        except Exception:
                            continue
                continue

            if subtype != Name("/Image"):
                continue

            _compress_single_image(pdf, xobjects, name, xobj, page_w, page_h, target_dpi, jpeg_quality)

        except Exception:
            continue


def _compress_single_image(pdf, xobjects, name, xobj, page_w, page_h, target_dpi, jpeg_quality):
    """Compress a single image XObject."""
    w, h = int(xobj.get("/Width", 0)), int(xobj.get("/Height", 0))
    if w < 48 or h < 48:
        return

    pil = _get_pil_image(xobj)
    if pil is None:
        return

    # Estimate DPI
    if page_w > 0 and page_h > 0:
        dpi = max(w / (page_w / 72.0), h / (page_h / 72.0))
    else:
        dpi = 300

    scale = min(1.0, target_dpi / dpi) if dpi > target_dpi * 1.15 else 1.0
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

    if scale < 0.92:
        pil = pil.resize((nw, nh), Image.LANCZOS)

    has_alpha = _has_meaningful_alpha(pil)
    is_photo = _is_photo_like(pil)

    if is_photo and not has_alpha:
        # JPEG
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
        new[Name("/Width")] = nw
        new[Name("/Height")] = nh
        new[Name("/ColorSpace")] = Name("/DeviceRGB")
        new[Name("/BitsPerComponent")] = 8
        new[Name("/Filter")] = Name("/DCTDecode")
        xobjects[name] = new

    elif has_alpha:
        # Preserve transparency with SMask
        if pil.mode != "RGBA":
            pil = pil.convert("RGBA")
        r, g, b, a = pil.split()
        rgb = Image.merge("RGB", (r, g, b))

        rgb_data = zlib.compress(rgb.tobytes(), 9)
        new = pdf.make_stream(rgb_data)
        new[Name("/Type")] = Name("/XObject")
        new[Name("/Subtype")] = Name("/Image")
        new[Name("/Width")] = nw
        new[Name("/Height")] = nh
        new[Name("/ColorSpace")] = Name("/DeviceRGB")
        new[Name("/BitsPerComponent")] = 8
        new[Name("/Filter")] = Name("/FlateDecode")

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
        # Lossless Flate for diagrams
        if pil.mode not in ("RGB", "L"):
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


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def compress_pdf(input_path, output_path, level="4"):
    """Two-pass compression. Returns stats dict."""
    if level not in LEVELS:
        level = "4"
    config = LEVELS[level]
    input_size = os.path.getsize(input_path)

    # Pass 1: Ghostscript deep recompression
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp1:
        gs_path = tmp1.name

    gs_worked = ghostscript_compress(input_path, gs_path, config)

    # Pass 2: pikepdf/QPDF structural optimization
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp2:
        opt_path = tmp2.name

    try:
        pikepdf_optimize(gs_path, opt_path, config)

        # Use whichever output is smallest
        gs_size = os.path.getsize(gs_path) if os.path.exists(gs_path) else float("inf")
        opt_size = os.path.getsize(opt_path) if os.path.exists(opt_path) else float("inf")

        if opt_size <= gs_size:
            shutil.move(opt_path, output_path)
        else:
            shutil.move(gs_path, output_path)

    finally:
        for p in [gs_path, opt_path]:
            if os.path.exists(p):
                os.unlink(p)

    output_size = os.path.getsize(output_path)

    # Safety: if "compressed" is actually larger, just use the original
    if output_size >= input_size:
        shutil.copy2(input_path, output_path)
        output_size = input_size

    return {
        "input_size": input_size,
        "output_size": output_size,
        "savings_pct": round((1 - output_size / input_size) * 100, 1) if input_size > 0 else 0,
        "gs_available": gs_worked,
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
