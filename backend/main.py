"""
英语作文批改 API 服务 (v3 - 百度手写OCR + MiMo语法分析 + 字符级坐标标注)
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
BAIDU_API_KEY = os.environ.get("BAIDU_API_KEY", "")
BAIDU_SECRET_KEY = os.environ.get("BAIDU_SECRET_KEY", "")
_baidu_token = {"access_token": None, "expires_at": 0}

MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "")
MIMO_BASE_URL = os.environ.get("MIMO_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")
MIMO_MODEL = os.environ.get("MIMO_MODEL", "mimo-v2-omni")

ERROR_COLORS = {
    'spelling': (255, 140, 0),
    'grammar': (220, 40, 40),
    'punctuation': (0, 120, 200),
    'style': (40, 160, 60),
}
CATEGORY_NAMES = {
    'spelling': '拼写',
    'grammar': '语法',
    'punctuation': '标点',
    'style': '表达',
}

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


# ====== Baidu OCR ======

def get_baidu_access_token():
    now = time.time()
    if _baidu_token["access_token"] and now < _baidu_token["expires_at"]:
        return _baidu_token["access_token"]
    url = (
        f"https://aip.baidubce.com/oauth/2.0/token"
        f"?grant_type=client_credentials"
        f"&client_id={BAIDU_API_KEY}"
        f"&client_secret={BAIDU_SECRET_KEY}"
    )
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
        "image": img_b64,
        "recognize_granularity": "small",
        "eng_granularity": "word",
        "language_type": "CHN_ENG",
        "probability": "true",
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    if "error_code" in data and data["error_code"] != 0:
        raise Exception(f"Baidu OCR error: {data.get('error_msg', data)}")
    lines = []
    for item in data.get("words_result", []):
        chars = [{"char": c["char"], "bbox": c["location"]} for c in item.get("chars", [])]
        lines.append({"text": item["words"], "bbox": item["location"], "chars": chars})
    return {"lines": lines}


# ====== MiMo Grammar Check ======

def check_grammar(text):
    """MiMo finds errors per OCR line. Each line is independent, no cross-line offset issues."""
    if not MIMO_API_KEY:
        return []

    # Split text into lines (matching OCR line boundaries)
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
        "max_tokens": 2000,
        "temperature": 0.1,
        "thinking": {"type": "disabled"},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{MIMO_BASE_URL}/chat/completions",
        data=body, method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MIMO_API_KEY}",
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"]

        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if not json_match:
            return []
        errors_raw = json.loads(json_match.group())

        # Find positions: each error is within its line
        errors = []
        for e in errors_raw:
            line_num = int(e.get("line", 1))
            error_phrase = e.get("error", "").strip()
            if not error_phrase or line_num < 1 or line_num > len(lines):
                continue

            line_text = lines[line_num - 1]
            line_lower = line_text.lower()

            # Find error in this line
            idx = line_lower.find(error_phrase.lower())
            if idx < 0:
                # Try partial match
                words = error_phrase.split()
                for n in range(min(3, len(words)), 0, -1):
                    phrase = ' '.join(words[:n])
                    idx = line_lower.find(phrase.lower())
                    if idx >= 0:
                        break
            if idx < 0:
                continue

            # Calculate global offset (sum of previous lines + newlines)
            global_offset = sum(len(lines[i]) + 1 for i in range(line_num - 1)) + idx

            errors.append({
                'message': e.get("message", ""),
                'replacements': [e.get("correction", "")],
                'offset': global_offset,
                'length': len(error_phrase),
                'line': line_num,
                'context': line_text[max(0, idx-10):idx+len(error_phrase)+10],
                'category': e.get("category", "grammar"),
                'rule_id': "mimo",
            })
        return errors
    except Exception as e:
        print(f"[ERROR] MiMo grammar check failed: {e}")
        return []


# ====== Coordinate Mapping ======

def build_char_map(ocr_lines):
    full_text = ""
    char_coords = []
    for line_idx, line in enumerate(ocr_lines):
        for ch_data in line["chars"]:
            char_coords.append({"char": ch_data["char"], "bbox": ch_data["bbox"], "line_idx": line_idx})
            full_text += ch_data["char"]
        if line_idx < len(ocr_lines) - 1:
            char_coords.append({"char": "\n", "bbox": None, "line_idx": line_idx})
            full_text += "\n"
    return full_text, char_coords


def map_errors_to_coords(errors, full_text, char_coords):
    """Map errors to coordinates. Adjusts offset for newlines not in char_coords."""
    # Count newline positions (MiMo sees them, but char_coords doesn't)
    nl_positions = [i for i, c in enumerate(full_text) if c == '\n']

    mapped = []
    for error in errors:
        offset = error["offset"]
        length = error["length"]

        # Adjust offset: char_coords doesn't have \n, so subtract newlines before offset
        nl_before = sum(1 for p in nl_positions if p < offset)
        adjusted_offset = offset - nl_before
        adjusted_length = min(length, len(char_coords) - adjusted_offset)

        if adjusted_offset < 0 or adjusted_length <= 0:
            continue

        char_bboxes = []
        for i in range(adjusted_offset, min(adjusted_offset + adjusted_length, len(char_coords))):
            cc = char_coords[i]
            if cc["bbox"] is not None:
                char_bboxes.append(cc["bbox"])
        if not char_bboxes:
            continue
        left = min(b["left"] for b in char_bboxes)
        top = min(b["top"] for b in char_bboxes)
        right = max(b["left"] + b["width"] for b in char_bboxes)
        bottom = max(b["top"] + b["height"] for b in char_bboxes)
        error_text = full_text[offset:offset+length].strip()
        mapped.append({
            **error,
            "error_text": error_text,
            "word_bbox": {"left": left, "top": top, "width": right - left, "height": bottom - top},
        })
    return mapped


# ====== Annotation ======

def annotate_image(image, mapped_errors):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
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
    return {"status": "ok", "version": "v3-mimo-grammar", "mimo": bool(MIMO_API_KEY)}


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

        # 1. Baidu OCR
        try:
            ocr_result = baidu_handwriting_ocr(contents)
        except Exception as e:
            results.append({
                'original_text': f'[OCR 失败: {e}]',
                'errors': [],
                'annotated_image': base64.b64encode(contents).decode('utf-8'),
            })
            continue

        ocr_lines = ocr_result["lines"]
        if not ocr_lines:
            results.append({'original_text': '', 'errors': [], 'annotated_image': base64.b64encode(contents).decode('utf-8')})
            continue

        # 2. Build full text + char coords
        full_text, char_coords = build_char_map(ocr_lines)
        if not full_text.strip():
            results.append({'original_text': '', 'errors': [], 'annotated_image': base64.b64encode(contents).decode('utf-8')})
            continue

        # 3. MiMo grammar check
        try:
            errors = check_grammar(full_text)
        except Exception as e:
            print(f"[ERROR] Grammar check: {e}")
            errors = []
        all_errors.extend(errors)

        # 4. Map to coordinates + annotate
        mapped_errors = map_errors_to_coords(errors, full_text, char_coords)
        annotated = annotate_image(image, mapped_errors)

        buffer = io.BytesIO()
        annotated.save(buffer, format='JPEG', quality=85)
        annotated_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        results.append({
            'original_text': full_text,
            'errors': [{
                'error': e.get('error_text', ''),
                'message': e['message'],
                'replacements': e.get('replacements', []),
                'category': e['category'],
            } for e in mapped_errors],
            'annotated_image': annotated_base64,
        })

    categories = {}
    for error in all_errors:
        cat = error['category']
        categories[cat] = categories.get(cat, 0) + 1

    return JSONResponse({
        'results': results,
        'summary': {'total_errors': len(all_errors), 'categories': categories},
    })


# ====== Static Files ======

@app.get("/")
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>Essay Grader v3</h1>")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
