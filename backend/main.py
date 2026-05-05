"""
英语作文批改 API 服务 (v4 - 百度OCR字符坐标 + MiMo语法分析)
"""
import os
import io
import re
import json
import time
import base64
import urllib.request
import urllib.parse
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from PIL import Image, ImageDraw, ImageFont

app = FastAPI(title="英语作文批改 API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

BAIDU_API_KEY = os.environ.get("BAIDU_API_KEY", "")
BAIDU_SECRET_KEY = os.environ.get("BAIDU_SECRET_KEY", "")
_baidu_token = {"access_token": None, "expires_at": 0}

MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "")
MIMO_BASE_URL = os.environ.get("MIMO_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")
MIMO_MODEL = os.environ.get("MIMO_MODEL", "mimo-v2-omni")

ERROR_COLORS = {'spelling': (255, 140, 0), 'grammar': (220, 40, 40),
                'punctuation': (0, 120, 200), 'style': (40, 160, 60),
                'translation': (220, 40, 40)}
CATEGORY_NAMES = {'spelling': '拼写', 'grammar': '语法', 'punctuation': '标点', 'style': '表达',
                   'translation': '翻译'}
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


# ====== Image pre-processing ======

def compress_image(image_bytes, max_size=800):
    """Compress image to reduce base64 size for Baidu OCR (limit 10MB)"""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=80)
    return buf.getvalue()


# ====== Baidu OCR ======

def get_baidu_access_token():
    now = time.time()
    if _baidu_token["access_token"] and now < _baidu_token["expires_at"]:
        return _baidu_token["access_token"]
    url = (f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials"
           f"&client_id={BAIDU_API_KEY}&client_secret={BAIDU_SECRET_KEY}")
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    if "access_token" not in data:
        raise Exception(f"Baidu token failed: {data}")
    _baidu_token["access_token"] = data["access_token"]
    _baidu_token["expires_at"] = now + data.get("expires_in", 2592000) - 60
    return _baidu_token["access_token"]


def baidu_handwriting_ocr(image_bytes):
    token = get_baidu_access_token()
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/handwriting?access_token={token}"
    body = urllib.parse.urlencode({
        "image": img_b64, "recognize_granularity": "small",
        "eng_granularity": "word", "language_type": "CHN_ENG", "probability": "true",
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    if "error_code" in data and data["error_code"] != 0:
        raise Exception(f"Baidu OCR error: {data.get('error_msg', data)}")
    lines = []
    for item in data.get("words_result", []):
        chars = [{"char": c["char"], "bbox": c["location"]} for c in item.get("chars", [])]
        lines.append({"text": item["words"], "bbox": item["location"], "chars": chars})
    return {"lines": lines}


# ====== MiMo Grammar Check (per-line) ======

def check_grammar(text):
    if not MIMO_API_KEY:
        return []
    lines = text.split('\n')
    prompt = (
        'You are an English writing assistant. Analyze the following lines of text '
        'for grammar, spelling, and punctuation errors.\n\n'
        'For EACH line, return errors with the LINE NUMBER (1-based).\n'
        'Return a JSON array. Each error must have:\n'
        '- "line": line number (1-based)\n'
        '- "error": ONLY the wrong word/phrase (max 3 words). Must appear in the line.\n'
        '- "correction": the corrected version\n'
        '- "category": one of "spelling", "grammar", "punctuation", "style"\n'
        '- "message": short Chinese explanation\n\n'
        'RULES:\n'
        '- "error" must be a short phrase (1-3 words) that actually appears in that line\n'
        '- Return ONLY the JSON array, no other text\n'
        '- If no errors, return []\n\n'
        'Lines:\n'
    )
    for i, line in enumerate(lines):
        prompt += f'Line {i+1}: {line}\n'

    body = json.dumps({
        "model": MIMO_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000, "temperature": 0.1,
        "thinking": {"type": "disabled"},
    }).encode("utf-8")
    req = urllib.request.Request(f"{MIMO_BASE_URL}/chat/completions", data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {MIMO_API_KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"]
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if not match:
            return []
        errors_raw = json.loads(match.group())

        errors = []
        for e in errors_raw:
            line_num = int(e.get("line", 1))
            error_phrase = e.get("error", "").strip()
            if not error_phrase or line_num < 1 or line_num > len(lines):
                continue
            # Verify error phrase exists in this line
            if error_phrase.lower() not in lines[line_num - 1].lower():
                continue
            errors.append({
                'message': e.get("message", ""),
                'replacements': [e.get("correction", "")],
                'category': e.get("category", "grammar"),
                'rule_id': "mimo",
                'line': line_num,
                'error_phrase': error_phrase,
            })
        return errors
    except Exception as e:
        print(f"[ERROR] MiMo grammar check failed: {e}")
        return []


# ====== Map errors to Baidu character coordinates ======

def find_phrase_in_chars(phrase, chars):
    """
    Find a phrase in the OCR character list and return the combined bbox.
    chars: [{"char": "H", "bbox": {"left":10, "top":20, "width":15, "height":25}}, ...]
    Returns: {"left": x, "top": y, "width": w, "height": h} or None
    """
    phrase_lower = phrase.lower()
    # Build char string from OCR chars
    char_str = ''.join(c["char"] for c in chars).lower()

    idx = char_str.find(phrase_lower)
    if idx < 0:
        # Try first word only
        first_word = phrase.split()[0] if phrase.split() else phrase
        idx = char_str.find(first_word.lower())
        if idx < 0:
            return None
        phrase_len = len(first_word)
    else:
        phrase_len = len(phrase)

    # Collect bboxes for matched characters
    bboxes = []
    for i in range(idx, min(idx + phrase_len, len(chars))):
        if chars[i].get("bbox"):
            bboxes.append(chars[i]["bbox"])

    if not bboxes:
        return None

    left = min(b["left"] for b in bboxes)
    top = min(b["top"] for b in bboxes)
    right = max(b["left"] + b["width"] for b in bboxes)
    bottom = max(b["top"] + b["height"] for b in bboxes)
    return {"left": left, "top": top, "width": right - left, "height": bottom - top}


def map_errors_to_coords(errors, ocr_lines):
    """Map MiMo errors to Baidu character coordinates using line number + text matching."""
    mapped = []
    for error in errors:
        line_num = error.get("line", 0)
        phrase = error.get("error_phrase", "")
        if not line_num or line_num < 1 or line_num > len(ocr_lines) or not phrase:
            continue

        ocr_line = ocr_lines[line_num - 1]
        bbox = find_phrase_in_chars(phrase, ocr_line["chars"])
        if bbox is None:
            continue

        mapped.append({
            **error,
            "error_text": phrase,
            "word_bbox": bbox,
        })
    return mapped


# ====== Annotation ======

def annotate_image(image, mapped_errors):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    # 优先使用中文字体
    cn_font_paths = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    font_path = None
    for p in cn_font_paths:
        if os.path.exists(p):
            font_path = p
            break
    try:
        if font_path:
            font = ImageFont.truetype(font_path, 16)
            small_font = ImageFont.truetype(font_path, 12)
        else:
            font = ImageFont.load_default()
            small_font = font
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    for error in mapped_errors:
        bbox = error["word_bbox"]
        color = ERROR_COLORS.get(error["category"], (128, 128, 128))
        x, y, w, h = bbox["left"], bbox["top"], bbox["width"], bbox["height"]
        line_y = y + h + 2
        draw.line([(x, line_y), (x + w, line_y)], fill=color, width=3)
        correction = error.get("replacements", [""])[0] if error.get("replacements") else ""
        if correction:
            draw.text((x, line_y + 3), f"-> {correction}", fill=color, font=small_font)
    return annotated


# ====== API ======

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "v4-direct-bbox", "mimo": bool(MIMO_API_KEY)}


@app.post("/api/grade")
async def grade_essay(files: List[UploadFile] = File(...)):
    if len(files) > 2:
        raise HTTPException(status_code=400, detail="最多只能上传2张图片")
    results = []
    all_errors = []

    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            results.append({'original_text': f'[图片读取失败: {e}]', 'errors': [], 'annotated_image': ''})
            continue

        # 1. Baidu OCR (compress large images first)
        try:
            compressed = compress_image(contents)
            ocr_result = baidu_handwriting_ocr(compressed)
        except Exception as e:
            results.append({'original_text': f'[OCR 失败: {e}]', 'errors': [],
                            'annotated_image': base64.b64encode(contents).decode('utf-8')})
            continue

        ocr_lines = ocr_result["lines"]
        if not ocr_lines:
            results.append({'original_text': '', 'errors': [],
                            'annotated_image': base64.b64encode(contents).decode('utf-8')})
            continue

        full_text = '\n'.join(line["text"] for line in ocr_lines)
        if not full_text.strip():
            results.append({'original_text': '', 'errors': [],
                            'annotated_image': base64.b64encode(contents).decode('utf-8')})
            continue

        # 2. MiMo grammar check
        try:
            errors = check_grammar(full_text)
        except Exception as e:
            print(f"[ERROR] Grammar check: {e}")
            errors = []
        all_errors.extend(errors)

        # 3. Map to Baidu character bboxes
        mapped_errors = map_errors_to_coords(errors, ocr_lines)

        # 4. Annotate
        annotated = annotate_image(image, mapped_errors)
        buf = io.BytesIO()
        annotated.save(buf, format='JPEG', quality=85)

        results.append({
            'original_text': full_text,
            'errors': [{'error': e.get('error_text', ''), 'message': e['message'],
                        'replacements': e.get('replacements', []), 'category': e['category'],
                        'line': e.get('line', 0)}
                       for e in mapped_errors],
            'annotated_image': base64.b64encode(buf.getvalue()).decode('utf-8'),
        })

    categories = {}
    for error in all_errors:
        cat = error['category']
        categories[cat] = categories.get(cat, 0) + 1
    return JSONResponse({'results': results, 'summary': {'total_errors': len(all_errors), 'categories': categories}})


# ====== Translation Check ======

def check_translation(text):
    """Use MiMo to check Chinese-English translation pairs"""
    if not MIMO_API_KEY:
        return []
    lines = text.split('\n')
    prompt = (
        'You are a translation checker. The following are vocabulary test answers '
        'with English words and their Chinese translations.\n\n'
        'Check each translation for correctness. For EACH line with an error:\n'
        '- "line": line number (1-based)\n'
        '- "error": the incorrect translation\n'
        '- "correction": the correct translation\n'
        '- "message": brief Chinese explanation\n\n'
        'RULES:\n'
        '- Only report actual translation errors, not minor differences\n'
        '- Return ONLY a JSON array\n'
        '- If no errors, return []\n\n'
        'Lines:\n'
    )
    for i, line in enumerate(lines):
        prompt += f'Line {i+1}: {line}\n'

    body = json.dumps({
        "model": MIMO_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000, "temperature": 0.1,
        "thinking": {"type": "disabled"},
    }).encode("utf-8")
    req = urllib.request.Request(f"{MIMO_BASE_URL}/chat/completions", data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {MIMO_API_KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"]
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if not match:
            return []
        errors_raw = json.loads(match.group())
        errors = []
        for e in errors_raw:
            line_num = int(e.get("line", 1))
            error_text = e.get("error", "").strip()
            if not error_text or line_num < 1 or line_num > len(lines):
                continue
            if error_text not in lines[line_num - 1]:
                continue
            errors.append({
                'message': e.get("message", ""),
                'replacements': [e.get("correction", "")],
                'category': 'translation',
                'rule_id': "mimo_translation",
                'line': line_num,
                'error_phrase': error_text,
            })
        return errors
    except Exception as e:
        print(f"[ERROR] MiMo translation check failed: {e}")
        return []


@app.post("/api/check-translation")
async def check_translation_endpoint(files: List[UploadFile] = File(...)):
    if len(files) > 2:
        raise HTTPException(status_code=400, detail="最多只能上传2张图片")
    results = []
    all_errors = []
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            results.append({'original_text': f'[图片读取失败: {e}]', 'errors': [], 'annotated_image': ''})
            continue
        try:
            compressed = compress_image(contents)
            ocr_result = baidu_handwriting_ocr(compressed)
        except Exception as e:
            results.append({'original_text': f'[OCR 失败: {e}]', 'errors': [],
                            'annotated_image': base64.b64encode(contents).decode('utf-8')})
            continue
        ocr_lines = ocr_result["lines"]
        if not ocr_lines:
            results.append({'original_text': '', 'errors': [],
                            'annotated_image': base64.b64encode(contents).decode('utf-8')})
            continue
        full_text = '\n'.join(line["text"] for line in ocr_lines)
        try:
            errors = check_translation(full_text)
        except Exception as e:
            print(f"[ERROR] Translation check: {e}")
            errors = []
        all_errors.extend(errors)
        mapped_errors = map_errors_to_coords(errors, ocr_lines)
        annotated = annotate_image(image, mapped_errors)
        buf = io.BytesIO()
        annotated.save(buf, format='JPEG', quality=85)
        results.append({
            'original_text': full_text,
            'errors': [{'error': e.get('error_text', ''), 'message': e['message'],
                        'replacements': e.get('replacements', []), 'category': e['category'],
                        'line': e.get('line', 0)} for e in mapped_errors],
            'annotated_image': base64.b64encode(buf.getvalue()).decode('utf-8'),
        })
    categories = {}
    for error in all_errors:
        categories[error['category']] = categories.get(error['category'], 0) + 1
    return JSONResponse({'results': results, 'summary': {'total_errors': len(all_errors), 'categories': categories}})


# ====== Static ======

@app.get("/")
async def serve_index():
    p = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(p) if os.path.exists(p) else HTMLResponse("<h1>Essay Grader v4</h1>")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
