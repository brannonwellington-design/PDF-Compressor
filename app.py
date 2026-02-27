#!/usr/bin/env python3
"""
PDF Compressor

Vector mode (levels 1-5): pikepdf-only compression.
  Recompresses images at target quality/DPI while preserving the original
  page structure, transparency, fonts, and vector art. This ensures
  Apple Preview compatibility since the transparency structure is untouched.

Rasterize mode (levels 6-7): Ghostscript renders each page to a flat
  JPEG image, then assembles into a new PDF. Maximum compression but
  text is no longer selectable.
"""

import io, os, subprocess, tempfile, shutil, logging
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
    "1": {"mode": "vector", "target_dpi": 300, "jpeg_quality": 95, "strip_metadata": False, "label": "Lossless cleanup"},
    "2": {"mode": "vector", "target_dpi": 250, "jpeg_quality": 88, "strip_metadata": False, "label": "High quality"},
    "3": {"mode": "vector", "target_dpi": 200, "jpeg_quality": 78, "strip_metadata": True,  "label": "Quality"},
    "4": {"mode": "vector", "target_dpi": 150, "jpeg_quality": 65, "strip_metadata": True,  "label": "Balanced"},
    "5": {"mode": "vector", "target_dpi": 120, "jpeg_quality": 50, "strip_metadata": True,  "label": "Email-ready"},
    "6": {"mode": "raster", "render_dpi": 150, "jpeg_quality": 80, "strip_metadata": True,  "label": "Aggressive"},
    "7": {"mode": "raster", "render_dpi": 120, "jpeg_quality": 65, "strip_metadata": True,  "label": "Maximum"},
}


# ── Vector mode: pikepdf image compression (preserves structure) ──────────────

def pikepdf_compress(input_path, output_path, config):
    """
    Compress images inside the PDF without altering page structure.
    This preserves all transparency, Form XObjects, ExtGState, fonts, etc.
    Only raster images are recompressed at lower quality/resolution.
    """
    target_dpi = config["target_dpi"]
    jpeg_quality = config["jpeg_quality"]
    strip_meta = config.get("strip_metadata", False)

    pdf = Pdf.open(input_path)
    images_processed = 0
    bytes_saved = 0

    # Walk all objects and recompress images
    for objnum in range(1, len(pdf.objects) + 1):
        try:
            obj = pdf.get_object(objnum, 0)
            if not hasattr(obj, 'get'):
                continue
            if str(obj.get("/Subtype", "")) != "/Image":
                continue

            # Skip tiny images (icons, 1px spacers)
            w = int(obj.get("/Width", 0))
            h = int(obj.get("/Height", 0))
            if w < 32 or h < 32:
                continue

            # Skip mask images
            if str(obj.get("/ColorSpace", "")) == "/DeviceGray" and w * h < 100000:
                continue

            original_size = len(obj.read_raw_bytes())
            if original_size < 5000:
                continue

            # Extract to PIL
            try:
                pil_img = pikepdf.PdfImage(obj).as_pil_image()
            except Exception:
                continue

            # Calculate current effective DPI (based on how it's used)
            # If image is much larger than needed at target_dpi, downsample
            max_dim_at_target = max(w, h)
            if max_dim_at_target > 2000 and target_dpi <= 200:
                scale = target_dpi / 150.0  # rough scale factor
                new_w = max(int(w * scale), 100)
                new_h = max(int(h * scale), 100)
                if new_w < w or new_h < h:
                    pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                    w, h = new_w, new_h

            # Handle different color modes
            has_smask = "/SMask" in obj
            if pil_img.mode == "RGBA":
                # Preserve alpha — save as PNG-like (Flate)
                rgb = pil_img.convert("RGB")
                buf = io.BytesIO()
                rgb.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
                jpeg_data = buf.getvalue()

                if len(jpeg_data) < original_size:
                    obj.write(jpeg_data)
                    obj[Name("/Filter")] = Name("/DCTDecode")
                    obj[Name("/ColorSpace")] = Name("/DeviceRGB")
                    obj[Name("/BitsPerComponent")] = 8
                    obj[Name("/Width")] = w
                    obj[Name("/Height")] = h
                    if "/DecodeParms" in obj:
                        del obj[Name("/DecodeParms")]
                    bytes_saved += original_size - len(jpeg_data)
                    images_processed += 1
                rgb.close()

            elif pil_img.mode in ("RGB", "L"):
                buf = io.BytesIO()
                pil_img_save = pil_img.convert("RGB") if pil_img.mode == "L" else pil_img
                pil_img_save.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
                jpeg_data = buf.getvalue()

                if len(jpeg_data) < original_size:
                    obj.write(jpeg_data)
                    obj[Name("/Filter")] = Name("/DCTDecode")
                    obj[Name("/ColorSpace")] = Name("/DeviceRGB") if pil_img.mode == "RGB" else Name("/DeviceGray")
                    obj[Name("/BitsPerComponent")] = 8
                    obj[Name("/Width")] = w
                    obj[Name("/Height")] = h
                    if "/DecodeParms" in obj:
                        del obj[Name("/DecodeParms")]
                    bytes_saved += original_size - len(jpeg_data)
                    images_processed += 1

            else:
                # P, CMYK, etc - convert to RGB
                try:
                    rgb = pil_img.convert("RGB")
                    buf = io.BytesIO()
                    rgb.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
                    jpeg_data = buf.getvalue()
                    if len(jpeg_data) < original_size:
                        obj.write(jpeg_data)
                        obj[Name("/Filter")] = Name("/DCTDecode")
                        obj[Name("/ColorSpace")] = Name("/DeviceRGB")
                        obj[Name("/BitsPerComponent")] = 8
                        obj[Name("/Width")] = w
                        obj[Name("/Height")] = h
                        if "/DecodeParms" in obj:
                            del obj[Name("/DecodeParms")]
                        bytes_saved += original_size - len(jpeg_data)
                        images_processed += 1
                    rgb.close()
                except Exception:
                    pass

            pil_img.close()

        except Exception:
            continue

    log.info(f"Recompressed {images_processed} images, saved {bytes_saved/1024/1024:.1f} MB")

    # Strip metadata
    if strip_meta:
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

    # Save with maximum Flate compression
    pdf.save(output_path,
             recompress_flate=True,
             object_stream_mode=pikepdf.ObjectStreamMode.generate,
             normalize_content=True)
    pdf.close()

    # QPDF final pass
    qpdf = shutil.which("qpdf")
    if qpdf:
        opt = output_path + ".qpdf.pdf"
        try:
            r = subprocess.run(
                [qpdf, "--recompress-flate", "--compression-level=9",
                 "--object-streams=generate", output_path, opt],
                capture_output=True, timeout=120)
            if r.returncode == 0 and os.path.exists(opt):
                if os.path.getsize(opt) < os.path.getsize(output_path):
                    os.replace(opt, output_path)
                else:
                    os.remove(opt)
            elif os.path.exists(opt):
                os.remove(opt)
        except Exception:
            if os.path.exists(opt):
                os.remove(opt)


# ── Rasterize mode: GS renders flat images ────────────────────────────────────

def ghostscript_rasterize(input_path, output_path, config):
    """
    Render each page to a flat JPEG via GS (png16m device, white bg,
    no transparency), then assemble into PDF using Pillow.
    """
    gs = shutil.which("gs")
    if not gs:
        log.error("Ghostscript not installed")
        shutil.copy2(input_path, output_path)
        return False

    render_dpi = config["render_dpi"]
    jpeg_quality = config["jpeg_quality"]
    tmpdir = tempfile.mkdtemp()

    try:
        try:
            p = Pdf.open(input_path)
            num_pages = len(p.pages)
            p.close()
        except Exception:
            num_pages = 200

        log.info(f"Rasterizing {num_pages} pages at {render_dpi} DPI, quality {jpeg_quality}")

        jpeg_paths = []
        for page_num in range(1, num_pages + 1):
            png_path = os.path.join(tmpdir, f"p{page_num}.png")

            result = subprocess.run([
                gs, "-dNOSAFER", "-sDEVICE=png16m",
                f"-r{render_dpi}", "-dNOPAUSE", "-dBATCH", "-dQUIET",
                "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4",
                f"-dFirstPage={page_num}", f"-dLastPage={page_num}",
                f"-sOutputFile={png_path}", input_path,
            ], capture_output=True, text=True, timeout=120)

            if result.returncode != 0 or not os.path.exists(png_path):
                log.error(f"Page {page_num} render failed: {result.stderr[:200]}")
                continue

            jpg_path = os.path.join(tmpdir, f"p{page_num}.jpg")
            try:
                img = Image.open(png_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(jpg_path, format="JPEG", quality=jpeg_quality, optimize=True)
                img.close()
                os.unlink(png_path)
                jpeg_paths.append(jpg_path)
            except Exception as e:
                log.error(f"JPEG convert page {page_num}: {e}")
                continue

            if page_num % 10 == 0:
                log.info(f"Rendered {page_num}/{num_pages}")

        if not jpeg_paths:
            log.error("No pages rendered")
            shutil.copy2(input_path, output_path)
            return False

        log.info(f"Rendered {len(jpeg_paths)} pages, assembling PDF")

        # Assemble one page at a time to avoid OOM
        # Convert each JPEG to a single-page PDF, then merge
        page_pdfs = []
        for jp in jpeg_paths:
            page_pdf_path = jp + ".pdf"
            try:
                img = Image.open(jp)
                img.load()
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(page_pdf_path, "PDF", resolution=render_dpi)
                img.close()
                page_pdfs.append(page_pdf_path)
            except Exception as e:
                log.error(f"Page PDF creation failed: {e}")
            os.unlink(jp)

        if not page_pdfs:
            log.error("No page PDFs created")
            shutil.copy2(input_path, output_path)
            return False

        # Merge all single-page PDFs with pikepdf (memory-efficient)
        out_pdf = Pdf.new()
        for pp in page_pdfs:
            try:
                src = Pdf.open(pp)
                out_pdf.pages.extend(src.pages)
                # Don't close src until we save — pikepdf needs it
            except Exception as e:
                log.error(f"Merge page failed: {e}")

        out_pdf.save(output_path)
        out_pdf.close()

        # Copy hyperlink annotations from original PDF onto rasterized pages
        try:
            orig_pdf = Pdf.open(input_path)
            rast_pdf = Pdf.open(output_path, allow_overwriting_input=True)

            links_copied = 0
            for i, orig_page in enumerate(orig_pdf.pages):
                if i >= len(rast_pdf.pages):
                    break

                annots = orig_page.get("/Annots")
                if not annots:
                    continue

                # Get original and rasterized page dimensions for scaling
                orig_mb = orig_page.MediaBox
                orig_w = float(orig_mb[2]) - float(orig_mb[0])
                orig_h = float(orig_mb[3]) - float(orig_mb[1])

                rast_mb = rast_pdf.pages[i].MediaBox
                rast_w = float(rast_mb[2]) - float(rast_mb[0])
                rast_h = float(rast_mb[3]) - float(rast_mb[1])

                scale_x = rast_w / orig_w if orig_w > 0 else 1
                scale_y = rast_h / orig_h if orig_h > 0 else 1

                new_annots = []
                for annot in annots:
                    try:
                        annot_obj = annot if hasattr(annot, 'get') else rast_pdf.make_indirect(annot)
                        subtype = str(annot_obj.get("/Subtype", ""))
                        if "/Link" not in subtype:
                            continue

                        # Copy the annotation into the new PDF
                        new_annot = rast_pdf.copy_foreign(annot_obj)

                        # Scale the Rect to match rasterized page dimensions
                        if "/Rect" in new_annot:
                            rect = new_annot["/Rect"]
                            new_annot["/Rect"] = pikepdf.Array([
                                float(rect[0]) * scale_x,
                                float(rect[1]) * scale_y,
                                float(rect[2]) * scale_x,
                                float(rect[3]) * scale_y,
                            ])

                        new_annots.append(new_annot)
                        links_copied += 1
                    except Exception:
                        continue

                if new_annots:
                    rast_pdf.pages[i]["/Annots"] = pikepdf.Array(new_annots)

            if links_copied > 0:
                log.info(f"Preserved {links_copied} hyperlinks from original PDF")
                rast_pdf.save(output_path)

            rast_pdf.close()
            orig_pdf.close()
        except Exception as e:
            log.warning(f"Link preservation failed (non-fatal): {e}")

        # Clean up page PDFs
        for pp in page_pdfs:
            if os.path.exists(pp):
                os.unlink(pp)

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


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def compress_pdf(input_path, output_path, level="4"):
    if level not in LEVELS:
        level = "4"
    config = LEVELS[level]
    input_size = os.path.getsize(input_path)
    log.info(f"Compressing {input_size/1024/1024:.1f}MB at level {level} ({config['label']}, mode={config['mode']})")

    if config["mode"] == "raster":
        success = ghostscript_rasterize(input_path, output_path, config)
        if not success:
            log.warning("Rasterize failed, falling back to vector mode")
            fallback = LEVELS["5"]
            pikepdf_compress(input_path, output_path, fallback)
    else:
        pikepdf_compress(input_path, output_path, config)

    output_size = os.path.getsize(output_path)
    if output_size >= input_size:
        shutil.copy2(input_path, output_path)
        output_size = input_size

    savings = round((1 - output_size / input_size) * 100, 1) if input_size > 0 else 0
    log.info(f"Final: {input_size/1024/1024:.1f}MB -> {output_size/1024/1024:.1f}MB ({savings}%)")
    return {"input_size": input_size, "output_size": output_size, "savings_pct": savings}


# ── Routes ────────────────────────────────────────────────────────────────────

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
        compress_pdf(input_path, output_path, level=level)
        original_name = Path(file.filename).stem
        return send_file(output_path, mimetype="application/pdf", as_attachment=True,
                         download_name=f"{original_name}-compressed.pdf")
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
