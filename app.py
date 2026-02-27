#!/usr/bin/env python3
"""
PDF Compressor — All-in-one web app.

Strategy for vector-heavy PDFs (Figma/Sketch exports, pitch decks):
  Ghostscript re-distills the entire PDF. This is the ONLY way to shrink
  Form XObjects (complex vector art from Figma/Sketch/Illustrator).

Strategy for image-heavy PDFs (scans, photo books):
  GS downsamples all images regardless of format, then pikepdf/QPDF
  does structural cleanup.
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

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("compress")

LEVELS = {
    "1": {"gs_pdfsettings": "/printer", "gs_dpi": 300, "gs_qfactor": 0.15, "strip_metadata": False, "label": "Lossless cleanup"},
    "2": {"gs_pdfsettings": "/printer", "gs_dpi": 250, "gs_qfactor": 0.20, "strip_metadata": False, "label": "High quality"},
    "3": {"gs_pdfsettings": "/ebook",   "gs_dpi": 200, "gs_qfactor": 0.40, "strip_metadata": True,  "label": "Quality"},
    "4": {"gs_pdfsettings": "/ebook",   "gs_dpi": 150, "gs_qfactor": 0.76, "strip_metadata": True,  "label": "Balanced"},
    "5": {"gs_pdfsettings": "/ebook",   "gs_dpi": 120, "gs_qfactor": 1.0,  "strip_metadata": True,  "label": "Email-ready"},
    "6": {"gs_pdfsettings": "/screen",  "gs_dpi": 96,  "gs_qfactor": 1.5,  "strip_metadata": True,  "label": "Aggressive"},
    "7": {"gs_pdfsettings": "/screen",  "gs_dpi": 72,  "gs_qfactor": 2.0,  "strip_metadata": True,  "label": "Maximum"},
}


def ghostscript_compress(input_path, output_path, config):
    gs = shutil.which("gs")
    if not gs:
        log.warning("Ghostscript not found")
        shutil.copy2(input_path, output_path)
        return False

    dpi = config["gs_dpi"]
    qfactor = config["gs_qfactor"]
    pdfsettings = config["gs_pdfsettings"]

    cmd = [
        gs,
        "-dNOSAFER",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.5",
        "-dNOPAUSE", "-dBATCH",
        f"-dPDFSETTINGS={pdfsettings}",

        f"-dColorImageResolution={dpi}",
        f"-dGrayImageResolution={dpi}",
        f"-dMonoImageResolution={min(dpi * 2, 300)}",
        "-dDownsampleColorImages=true",
        "-dDownsampleGrayImages=true",
        "-dDownsampleMonoImages=true",
        "-dColorImageDownsampleType=/Bicubic",
        "-dGrayImageDownsampleType=/Bicubic",
        "-dMonoImageDownsampleType=/Subsample",
        "-dColorImageDownsampleThreshold=1.0",
        "-dGrayImageDownsampleThreshold=1.0",
        "-dMonoImageDownsampleThreshold=1.0",

        "-dAutoFilterColorImages=false",
        "-dAutoFilterGrayImages=false",
        "-dColorImageFilter=/DCTEncode",
        "-dGrayImageFilter=/DCTEncode",

        "-dPrinted=false",
        "-dPreserveAnnots=true",
        "-dSubsetFonts=true",
        "-dEmbedAllFonts=true",
        "-dCompressFonts=true",
        "-dCompressPages=true",
        "-dUseFlateCompression=true",
        "-dOptimize=true",
        "-dDetectDuplicateImages=true",
        "-dConvertCMYKImagesToRGB=true",

        "-c",
        f"<< /ColorACSImageDict << /QFactor {qfactor} /Blend 1 /HSamples [1 1 1 1] /VSamples [1 1 1 1] >> "
        f"/GrayACSImageDict << /QFactor {qfactor} /Blend 1 /HSamples [1 1 1 1] /VSamples [1 1 1 1] >> "
        f"/ColorConversionStrategy /sRGB >> setdistillerparams",
        "-f",

        f"-sOutputFile={output_path}",
        input_path,
    ]

    try:
        log.info(f"GS: dpi={dpi}, qfactor={qfactor}, pdfsettings={pdfsettings}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            log.error(f"GS failed (rc={result.returncode})")
            log.error(f"GS stderr: {result.stderr[:1000]}")
            log.error(f"GS stdout: {result.stdout[:1000]}")
            if os.path.exists(output_path):
                os.unlink(output_path)
            shutil.copy2(input_path, output_path)
            return False

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            in_sz = os.path.getsize(input_path)
            out_sz = os.path.getsize(output_path)
            log.info(f"GS: {in_sz/1024/1024:.1f}MB -> {out_sz/1024/1024:.1f}MB ({(1-out_sz/in_sz)*100:.0f}%)")
            return True

        log.error("GS produced empty output")
        shutil.copy2(input_path, output_path)
        return False

    except subprocess.TimeoutExpired:
        log.error("GS timed out (600s)")
        if os.path.exists(output_path):
            os.unlink(output_path)
        shutil.copy2(input_path, output_path)
        return False
    except Exception as e:
        log.error(f"GS exception: {e}")
        if os.path.exists(output_path):
            os.unlink(output_path)
        shutil.copy2(input_path, output_path)
        return False


def pikepdf_optimize(input_path, output_path, config):
    try:
        pdf = Pdf.open(input_path)

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


def compress_pdf(input_path, output_path, level="4"):
    if level not in LEVELS:
        level = "4"
    config = LEVELS[level]
    input_size = os.path.getsize(input_path)

    log.info(f"Compressing {input_size/1024/1024:.1f}MB at level {level}")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp1:
        gs_path = tmp1.name
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp2:
        opt_path = tmp2.name

    try:
        gs_worked = ghostscript_compress(input_path, gs_path, config)
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
        "gs_available": gs_worked,
    }


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
    return jsonify({
        "status": "ok",
        "ghostscript": gs_ver or ("found" if gs else "NOT INSTALLED"),
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
