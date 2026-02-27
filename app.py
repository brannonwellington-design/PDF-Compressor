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
    Memory-efficient rasterize: one page at a time.
    Renders to PNG with pngalpha device (preserves transparency),
    embeds as Flate-compressed RGB + alpha SMask in the output PDF.
    """
    gs = shutil.which("gs")
    if not gs:
        log.warning("Ghostscript not found for rasterize")
        shutil.copy2(input_path, output_path)
        return False

    render_dpi = config["render_dpi"]
    tmpdir = tempfile.mkdtemp()

    try:
        # Count pages
        try:
            p = Pdf.open(input_path)
            num_pages = len(p.pages)
            p.close()
        except Exception:
            num_pages = 100

        log.info(f"Rasterizing {num_pages} pages at {render_dpi} DPI (preserving transparency)")

        out_pdf = Pdf.new()

        for page_num in range(1, num_pages + 1):
            png_path = os.path.join(tmpdir, f"p{page_num}.png")

            # Render single page to PNG with alpha channel
            render_cmd = [
                gs, "-dNOSAFER", "-sDEVICE=pngalpha",
                f"-r{render_dpi}",
                "-dNOPAUSE", "-dBATCH", "-dQUIET",
                "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4",
                f"-dFirstPage={page_num}", f"-dLastPage={page_num}",
                f"-sOutputFile={png_path}",
                input_path,
            ]

            result = subprocess.run(render_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0 or not os.path.exists(png_path):
                log.error(f"Failed to render page {page_num}: {result.stderr[:200]}")
                continue

            try:
                img = Image.open(png_path)
                img_w, img_h = img.size
                has_alpha = img.mode == "RGBA"

                import zlib

                if has_alpha:
                    # Split into RGB + Alpha
                    r, g, b, a = img.split()
                    rgb = Image.merge("RGB", (r, g, b))

                    # Compress RGB data
                    rgb_data = zlib.compress(rgb.tobytes(), 6)

                    # Compress alpha data
                    alpha_data = zlib.compress(a.tobytes(), 6)

                    rgb.close()
                else:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    rgb_data = zlib.compress(img.tobytes(), 6)
                    alpha_data = None

                img.close()
            except Exception as e:
                log.error(f"Image processing error page {page_num}: {e}")
                if os.path.exists(png_path):
                    os.unlink(png_path)
                continue

            # Remove PNG immediately
            if os.path.exists(png_path):
                os.unlink(png_path)

            # Calculate page size in points
            page_w_pt = img_w * 72.0 / render_dpi
            page_h_pt = img_h * 72.0 / render_dpi

            # Build image stream
            img_stream = out_pdf.make_stream(rgb_data)
            img_stream[Name("/Type")] = Name("/XObject")
            img_stream[Name("/Subtype")] = Name("/Image")
            img_stream[Name("/Width")] = img_w
            img_stream[Name("/Height")] = img_h
            img_stream[Name("/ColorSpace")] = Name("/DeviceRGB")
            img_stream[Name("/BitsPerComponent")] = 8
            img_stream[Name("/Filter")] = Name("/FlateDecode")

            # Add alpha mask if present
            if alpha_data is not None:
                smask = out_pdf.make_stream(alpha_data)
                smask[Name("/Type")] = Name("/XObject")
                smask[Name("/Subtype")] = Name("/Image")
                smask[Name("/Width")] = img_w
                smask[Name("/Height")] = img_h
                smask[Name("/ColorSpace")] = Name("/DeviceGray")
                smask[Name("/BitsPerComponent")] = 8
                smask[Name("/Filter")] = Name("/FlateDecode")
                img_stream[Name("/SMask")] = smask

            # Build page
            page_dict = pikepdf.Dictionary({
                Name("/Type"): Name("/Page"),
                Name("/MediaBox"): pikepdf.Array([0, 0, page_w_pt, page_h_pt]),
                Name("/Resources"): pikepdf.Dictionary({
                    Name("/XObject"): pikepdf.Dictionary({
                        Name("/Im0"): img_stream,
                    }),
                }),
            })

            content = f"q {page_w_pt:.4f} 0 0 {page_h_pt:.4f} 0 0 cm /Im0 Do Q"
            page_dict[Name("/Contents")] = out_pdf.make_stream(content.encode())

            out_pdf.pages.append(page_dict)

            if page_num % 10 == 0:
                log.info(f"Processed {page_num}/{num_pages} pages")

        if len(out_pdf.pages) == 0:
            log.error("No pages rendered successfully")
            out_pdf.close()
            shutil.copy2(input_path, output_path)
            return False

        log.info(f"Assembled {len(out_pdf.pages)} pages")
        out_pdf.save(output_path)
        out_pdf.close()

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
