#!/usr/bin/env python3
"""
PDF Compressor — Two strategies:

1. Vector-preserving (Levels 1–5): Ghostscript re-distills the PDF,
   compressing images but keeping vectors as vectors. Good when you
   need to preserve text selectability and vector crispness.

2. Rasterizing (Levels 6–7): Renders each page to a JPEG image at a
   target DPI, then assembles a new PDF from those images. This is the
   ONLY way to dramatically shrink vector-heavy PDFs (Figma, Sketch,
   InDesign exports) because it eliminates all the complex path data.
   Text is no longer selectable but the file gets MUCH smaller.
"""

import io
import os
import subprocess
import sys
import tempfile
import shutil
import logging
from pathlib import Path

from flask import Flask, request, send_file, render_template, jsonify

import pikepdf
from pikepdf import Pdf, Name
from PIL import Image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("compress")

LEVELS = {
    "1": {"mode": "vector", "gs_pdfsettings": "/printer", "gs_dpi": 300, "gs_qfactor": 0.15, "strip_metadata": False, "label": "Lossless cleanup"},
    "2": {"mode": "vector", "gs_pdfsettings": "/printer", "gs_dpi": 250, "gs_qfactor": 0.20, "strip_metadata": False, "label": "High quality"},
    "3": {"mode": "vector", "gs_pdfsettings": "/ebook",   "gs_dpi": 200, "gs_qfactor": 0.40, "strip_metadata": True,  "label": "Quality"},
    "4": {"mode": "vector", "gs_pdfsettings": "/ebook",   "gs_dpi": 150, "gs_qfactor": 0.76, "strip_metadata": True,  "label": "Balanced"},
    "5": {"mode": "vector", "gs_pdfsettings": "/screen",  "gs_dpi": 120, "gs_qfactor": 1.0,  "strip_metadata": True,  "label": "Email-ready"},
    "6": {"mode": "raster", "render_dpi": 150, "jpeg_quality": 80, "strip_metadata": True,  "label": "Aggressive"},
    "7": {"mode": "raster", "render_dpi": 120, "jpeg_quality": 65, "strip_metadata": True,  "label": "Maximum"},
}


# ─── Strategy 1: Vector-preserving GS compression ────────────────────────────

def ghostscript_vector_compress(input_path, output_path, config):
    gs = shutil.which("gs")
    if not gs:
        log.warning("Ghostscript not found")
        shutil.copy2(input_path, output_path)
        return False

    dpi = config["gs_dpi"]
    qfactor = config["gs_qfactor"]
    pdfsettings = config["gs_pdfsettings"]

    cmd = [
        gs, "-dNOSAFER", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.5",
        "-dNOPAUSE", "-dBATCH",
        f"-sOutputFile={output_path}",
        f"-dPDFSETTINGS={pdfsettings}",

        f"-dColorImageResolution={dpi}",
        f"-dGrayImageResolution={dpi}",
        f"-dMonoImageResolution={min(dpi * 2, 300)}",
        "-dDownsampleColorImages=true", "-dDownsampleGrayImages=true",
        "-dDownsampleMonoImages=true",
        "-dColorImageDownsampleType=/Bicubic",
        "-dGrayImageDownsampleType=/Bicubic",
        "-dColorImageDownsampleThreshold=1.0",
        "-dGrayImageDownsampleThreshold=1.0",

        "-dAutoFilterColorImages=false", "-dAutoFilterGrayImages=false",
        "-dColorImageFilter=/DCTEncode", "-dGrayImageFilter=/DCTEncode",

        "-dPrinted=false", "-dPreserveAnnots=true",
        "-dSubsetFonts=true", "-dEmbedAllFonts=true", "-dCompressFonts=true",
        "-dCompressPages=true", "-dUseFlateCompression=true",
        "-dOptimize=true", "-dDetectDuplicateImages=true",
        "-dConvertCMYKImagesToRGB=true",

        "-c",
        f"<< /ColorACSImageDict << /QFactor {qfactor} /Blend 1 /HSamples [1 1 1 1] /VSamples [1 1 1 1] >> "
        f"/GrayACSImageDict << /QFactor {qfactor} /Blend 1 /HSamples [1 1 1 1] /VSamples [1 1 1 1] >> "
        f"/ColorConversionStrategy /sRGB >> setdistillerparams",
        "-f",

        input_path,
    ]

    try:
        log.info(f"GS vector: dpi={dpi}, qfactor={qfactor}, pdfsettings={pdfsettings}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            log.error(f"GS failed (rc={result.returncode}): {result.stderr[:500]}")
            if os.path.exists(output_path):
                os.unlink(output_path)
            shutil.copy2(input_path, output_path)
            return False

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            in_sz = os.path.getsize(input_path)
            out_sz = os.path.getsize(output_path)
            log.info(f"GS vector: {in_sz/1024/1024:.1f}MB -> {out_sz/1024/1024:.1f}MB ({(1-out_sz/in_sz)*100:.0f}%)")
            return True

        shutil.copy2(input_path, output_path)
        return False

    except Exception as e:
        log.error(f"GS exception: {e}")
        if os.path.exists(output_path):
            os.unlink(output_path)
        shutil.copy2(input_path, output_path)
        return False


# ─── Strategy 2: Rasterize pages for maximum compression ─────────────────────

def ghostscript_rasterize(input_path, output_path, config):
    """
    Render each page to a PNG image at target DPI (to preserve transparency
    and color accuracy), then convert to JPEG and assemble into a PDF.
    PNG rendering avoids the black-rectangle artifacts from direct JPEG rendering.
    """
    gs = shutil.which("gs")
    if not gs:
        log.warning("Ghostscript not found for rasterize")
        shutil.copy2(input_path, output_path)
        return False

    render_dpi = config["render_dpi"]
    jpeg_quality = config["jpeg_quality"]

    tmpdir = tempfile.mkdtemp()

    try:
        # Step 1: Render all pages to PNG (not JPEG — avoids transparency issues)
        log.info(f"Rasterizing at {render_dpi} DPI")
        render_cmd = [
            gs, "-dNOSAFER", "-sDEVICE=png16m",
            f"-r{render_dpi}",
            "-dNOPAUSE", "-dBATCH",
            "-dTextAlphaBits=4",
            "-dGraphicsAlphaBits=4",
            f"-sOutputFile={tmpdir}/page_%04d.png",
            input_path,
        ]

        result = subprocess.run(render_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            log.error(f"GS render failed: {result.stderr[:500]}")
            shutil.copy2(input_path, output_path)
            return False

        # Step 2: Collect rendered pages
        page_files = sorted([
            os.path.join(tmpdir, f)
            for f in os.listdir(tmpdir)
            if f.startswith("page_") and f.endswith(".png")
        ])

        if not page_files:
            log.error("No pages rendered")
            shutil.copy2(input_path, output_path)
            return False

        log.info(f"Rendered {len(page_files)} pages as PNG")

        # Step 3: Convert PNGs to JPEG-compressed PDF pages using Pillow
        # Open all pages as PIL images, convert to RGB (flatten any alpha onto white)
        pil_pages = []
        for pf in page_files:
            img = Image.open(pf)
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")
            pil_pages.append(img)

        if not pil_pages:
            log.error("No PIL images created")
            shutil.copy2(input_path, output_path)
            return False

        # Step 4: Save as multi-page PDF using Pillow
        # Pillow handles page sizing and image embedding correctly
        first_page = pil_pages[0]
        rest = pil_pages[1:] if len(pil_pages) > 1 else []

        first_page.save(
            output_path,
            "PDF",
            resolution=render_dpi,
            save_all=True,
            append_images=rest,
            quality=jpeg_quality,
            optimize=True,
        )

        # Close all images
        for img in pil_pages:
            img.close()

        out_sz = os.path.getsize(output_path)
        in_sz = os.path.getsize(input_path)
        log.info(f"Rasterize: {in_sz/1024/1024:.1f}MB -> {out_sz/1024/1024:.1f}MB ({(1-out_sz/in_sz)*100:.0f}%)")
        return True

    except Exception as e:
        log.error(f"Rasterize error: {e}")
        import traceback
        log.error(traceback.format_exc())
        shutil.copy2(input_path, output_path)
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─── pikepdf structural optimization ─────────────────────────────────────────

def pikepdf_optimize(input_path, output_path, config):
    try:
        pdf = Pdf.open(input_path)

        if config.get("strip_metadata", False):
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

        pdf.remove_unreferenced_resources()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        pdf.save(tmp_path, recompress_flate=True,
                 object_stream_mode=pikepdf.ObjectStreamMode.generate,
                 normalize_content=True)
        pdf.close()

        qpdf = shutil.which("qpdf")
        if qpdf:
            opt = tmp_path + ".opt.pdf"
            r = subprocess.run(
                [qpdf, "--recompress-flate", "--compression-level=9",
                 "--object-streams=generate", tmp_path, opt],
                capture_output=True, timeout=120)
            if r.returncode == 0 and os.path.exists(opt):
                if os.path.getsize(opt) < os.path.getsize(tmp_path):
                    os.replace(opt, tmp_path)
                else:
                    os.remove(opt)
            elif os.path.exists(opt):
                os.remove(opt)

        shutil.move(tmp_path, output_path)

    except Exception as e:
        log.error(f"pikepdf error: {e}")
        shutil.copy2(input_path, output_path)


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def compress_pdf(input_path, output_path, level="4"):
    if level not in LEVELS:
        level = "4"
    config = LEVELS[level]
    input_size = os.path.getsize(input_path)

    log.info(f"Compressing {input_size/1024/1024:.1f}MB at level {level} ({config['label']}, mode={config['mode']})")

    if config["mode"] == "raster":
        # Rasterize mode: render pages to JPEG images
        gs_worked = ghostscript_rasterize(input_path, output_path, config)

        if not gs_worked:
            # Fallback to vector compress
            log.warning("Rasterize failed, falling back to vector mode")
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                fallback_config = LEVELS["5"]  # Use email-ready as fallback
                ghostscript_vector_compress(input_path, tmp_path, fallback_config)
                pikepdf_optimize(tmp_path, output_path, fallback_config)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    else:
        # Vector-preserving mode: GS re-distill + pikepdf cleanup
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp1:
            gs_path = tmp1.name
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp2:
            opt_path = tmp2.name

        try:
            gs_worked = ghostscript_vector_compress(input_path, gs_path, config)
            pikepdf_optimize(gs_path, opt_path, config)

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
    if output_size >= input_size:
        shutil.copy2(input_path, output_path)
        output_size = input_size

    savings = round((1 - output_size / input_size) * 100, 1) if input_size > 0 else 0
    log.info(f"Final: {input_size/1024/1024:.1f}MB -> {output_size/1024/1024:.1f}MB ({savings}%)")

    return {
        "input_size": input_size,
        "output_size": output_size,
        "savings_pct": savings,
    }


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    gs = shutil.which("gs")
    qpdf = shutil.which("qpdf")
    gs_ver = ""
    if gs:
        try:
            r = subprocess.run([gs, "--version"], capture_output=True, text=True, timeout=5)
            gs_ver = r.stdout.strip()
        except:
            pass

    # Quick test: can GS actually render a page to JPEG?
    gs_jpeg_works = False
    gs_pdfwrite_works = False
    if gs:
        try:
            # Create a tiny 1-page PDF
            test_pdf = Pdf.new()
            page = pikepdf.Dictionary({
                Name("/Type"): Name("/Page"),
                Name("/MediaBox"): pikepdf.Array([0, 0, 72, 72]),
                Name("/Contents"): test_pdf.make_stream(b""),
            })
            test_pdf.pages.append(page)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                test_pdf.save(tf.name)
                test_in = tf.name
            test_pdf.close()

            # Test jpeg device
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tj:
                test_jpg = tj.name
            r = subprocess.run(
                [gs, "-dNOSAFER", "-sDEVICE=jpeg", "-r72", "-dNOPAUSE", "-dBATCH",
                 f"-sOutputFile={test_jpg}", test_in],
                capture_output=True, text=True, timeout=10)
            gs_jpeg_works = r.returncode == 0 and os.path.exists(test_jpg) and os.path.getsize(test_jpg) > 0
            gs_jpeg_err = r.stderr[:200] if r.returncode != 0 else ""

            # Test pdfwrite device
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tp:
                test_out = tp.name
            r2 = subprocess.run(
                [gs, "-dNOSAFER", "-sDEVICE=pdfwrite", "-dNOPAUSE", "-dBATCH",
                 f"-sOutputFile={test_out}", test_in],
                capture_output=True, text=True, timeout=10)
            gs_pdfwrite_works = r2.returncode == 0

            for f in [test_in, test_jpg, test_out]:
                if os.path.exists(f):
                    os.unlink(f)

        except Exception as e:
            gs_jpeg_err = str(e)

    return jsonify({
        "status": "ok",
        "ghostscript": gs_ver or ("found" if gs else "NOT INSTALLED"),
        "gs_jpeg_device": "works" if gs_jpeg_works else f"BROKEN: {gs_jpeg_err if 'gs_jpeg_err' in dir() else 'unknown'}",
        "gs_pdfwrite_device": "works" if gs_pdfwrite_works else "BROKEN",
        "qpdf": "installed" if qpdf else "NOT INSTALLED",
    })

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
            output_path, mimetype="application/pdf", as_attachment=True,
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
