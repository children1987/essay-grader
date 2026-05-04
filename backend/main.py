"""
英语作文批改 API 服务
MiMo V2.5 错误分析 + 行号定位 + 水平投影行检测
"""
import os
import io
import re
import json
import base64
import traceback
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from PIL import Image, ImageDraw, ImageFont
import requests

app = FastAPI(title="英语作文批改 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MIMO_API_URL = "https://token-plan-cn.xiaomimimo.com/v1/chat/completions"
MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "")
MIMO_MODEL = "mimo-v2-omni"

ERROR_COLORS = {
    "spelling": (255, 140, 0),
    "grammar": (220, 40, 40),
    "punctuation": (0, 120, 200),
    "style": (40, 160, 60),
    "translation": (180, 0, 180),
}


def resize_image(image: Image.Image, max_width: int = 800) -> Image.Image:
    w, h = image.size
    if w <= max_width:
        return image
    return image.resize((max_width, int(h * max_width / w)), Image.LANCZOS)


def image_to_base64(image: Image.Image, quality: int = 70) -> str:
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _load_fonts():
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(p):
            try:
                return (ImageFont.truetype(p, 16), ImageFont.truetype(p, 13), ImageFont.truetype(p, 14))
            except Exception:
                pass
    f = ImageFont.load_default()
    return f, f, f


# ========== 行检测 ==========

def detect_text_lines(image: Image.Image, expected_count: int = 0) -> list:
    """水平投影分析检测文字行中心 y 坐标，自动找到作文正文区域"""
    gray = image.convert("L")
    w, h = gray.size
    pixels = gray.load()
    threshold = 140
    projection = []
    for y in range(h):
        count = sum(1 for x in range(0, w, 3) if pixels[x, y] < threshold)
        projection.append(count)

    # Moving average
    smooth = []
    for y in range(h):
        start = max(0, y - 3)
        end = min(h, y + 4)
        smooth.append(sum(projection[start:end]) / (end - start))

    # Find peaks
    avg = sum(smooth) / len(smooth) if smooth else 1
    peaks = []
    in_peak = False
    peak_start = 0
    for y in range(h):
        if smooth[y] > avg * 0.3 and not in_peak:
            in_peak = True
            peak_start = y
        elif smooth[y] <= avg * 0.3 and in_peak:
            in_peak = False
            peaks.append((peak_start + y) // 2)

    # 过滤掉太短的间隔（合并靠得很近的行）
    if len(peaks) > 1:
        merged = [peaks[0]]
        for p in peaks[1:]:
            if p - merged[-1] < 30:  # 间距小于30px视为同一行
                merged[-1] = (merged[-1] + p) // 2
            else:
                merged.append(p)
        peaks = merged

    if len(peaks) <= 1:
        return peaks

    # === 找到作文正文区域（行间距最均匀的连续子序列）===
    if expected_count > 0 and len(peaks) > expected_count:
        # 如果已知期望行数，找到最匹配的子序列
        best_score = float('inf')
        best_slice = peaks
        for start_idx in range(len(peaks) - expected_count + 1):
            candidate = peaks[start_idx:start_idx + expected_count]
            # 计算间距的方差（越小越均匀）
            spacings = [candidate[i+1] - candidate[i] for i in range(len(candidate)-1)]
            if not spacings:
                continue
            mean_s = sum(spacings) / len(spacings)
            variance = sum((s - mean_s) ** 2 for s in spacings) / len(spacings)
            if variance < best_score:
                best_score = variance
                best_slice = candidate
        print(f"  Essay lines: indices {peaks.index(best_slice[0])}-{peaks.index(best_slice[-1])}, variance={best_score:.0f}")
        return best_slice

    # 无期望行数时：找最长的均匀子序列
    # 对每个可能的起止范围，计算间距方差
    best_len = 0
    best_slice = peaks
    best_variance = float('inf')
    for start_idx in range(len(peaks)):
        for end_idx in range(start_idx + 3, len(peaks) + 1):
            candidate = peaks[start_idx:end_idx]
            spacings = [candidate[i+1] - candidate[i] for i in range(len(candidate)-1)]
            if not spacings:
                continue
            mean_s = sum(spacings) / len(spacings)
            # 跳过间距太小（<15px）或太大（>100px）的
            if mean_s < 15 or mean_s > 100:
                continue
            variance = sum((s - mean_s) ** 2 for s in spacings) / len(spacings)
            # 方差要小于间距的30%才算均匀
            if variance > (mean_s * 0.3) ** 2:
                continue
            if len(candidate) > best_len or (len(candidate) == best_len and variance < best_variance):
                best_len = len(candidate)
                best_slice = candidate
                best_variance = variance

    print(f"  Essay region: {best_len} lines, spacing variance={best_variance:.0f}")
    return best_slice


def detect_line_x_range(image: Image.Image, y_center: int) -> tuple:
    """水平投影检测某行文字的左右边界 (x_start, x_end)"""
    gray = image.convert("L")
    w, h = gray.size
    pixels = gray.load()
    threshold = 140
    band = 12  # 上下扫描范围
    margin = int(w * 0.08)  # 跳过左侧8%（页边红线区域）

    # 垂直投影：每列有多少暗像素（跳过左侧页边线）
    projection = [0] * margin  # 前 margin 列设为0
    for x in range(margin, w):
        count = sum(1 for y in range(max(0, y_center - band), min(h, y_center + band))
                    if pixels[x, y] < threshold)
        projection.append(count)

    # 找到有文字的列范围（投影 > 0 的区域）
    avg = sum(projection) / len(projection) if projection else 0
    text_cols = [i for i, v in enumerate(projection) if v > avg * 0.1]

    if len(text_cols) < 5:
        return (int(w * 0.08), int(w * 0.95))

    return (text_cols[0], text_cols[-1])


def compute_x_from_ocr(error_text: str, ocr_line: str) -> tuple:
    """在OCR行文本中查找错误词，返回 (start_pct, width_pct)
    start_pct: 错误词起始位置占行长度的百分比 (0-100)
    width_pct: 错误词宽度占行长度的百分比 (0-100)
    """
    if not ocr_line or not error_text:
        return (50.0, 10.0)

    error_clean = ' '.join(error_text.split())
    line_lower = ocr_line.lower()
    line_len = len(ocr_line)
    if line_len == 0:
        return (50.0, 10.0)

    # 1. 精确匹配错误短语
    idx = line_lower.find(error_clean.lower())
    if idx >= 0:
        start_pct = (idx / line_len) * 100
        width_pct = (len(error_clean) / line_len) * 100
        return (start_pct, width_pct)

    # 2. 匹配错误短语的前几个词（MiMo可能OCR有细微差异）
    words = error_clean.split()
    for n_words in range(min(3, len(words)), 0, -1):
        phrase = ' '.join(words[:n_words])
        idx = line_lower.find(phrase.lower())
        if idx >= 0:
            start_pct = (idx / line_len) * 100
            # 估算完整短语宽度
            est_len = len(error_clean) if n_words == len(words) else len(phrase) + 3
            width_pct = (est_len / line_len) * 100
            return (start_pct, width_pct)

    # 3. 只匹配第一个词
    if words:
        idx = line_lower.find(words[0].lower())
        if idx >= 0:
            start_pct = (idx / line_len) * 100
            width_pct = (len(error_clean) / line_len) * 100
            return (start_pct, width_pct)

    # 4. fallback: 用MiMo的x_pct（会在annotate_image里传入）
    return (None, None)  # 表示未匹配，fallback到MiMo x_pct


# ========== MiMo ==========

def mimo_analyze_essay(image: Image.Image) -> dict:
    img_b64 = image_to_base64(image)

    prompt = """Analyze this English essay image. Find ALL grammar/spelling/punctuation errors.

Output strict JSON only:
{"text":"full text","errors":[{"error":"wrong word","correction":"correct word","category":"grammar|spelling|punctuation|style","message":"Chinese explanation","line":3,"x_pct":30}]}

Rules:
- "error" = ONLY the specific wrong word/phrase (max 3 words). NOT the full sentence.
- "correction" = the corrected version of just that word/phrase
- line = 1-based line number (essay body only)
- x_pct = word start position (0-100 from left)
- Find every error including subject-verb agreement, tense
- Output ONLY JSON"""

    payload = {
        "model": MIMO_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": prompt}
        ]}],
        "max_tokens": 2000,
        "temperature": 0.1,
        "thinking": {"type": "disabled"},
    }

    try:
        resp = requests.post(MIMO_API_URL,
            headers={"Authorization": f"Bearer {MIMO_API_KEY}", "Content-Type": "application/json"},
            json=payload, timeout=120)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        if not content:
            reasoning = resp.json()["choices"][0]["message"].get("reasoning_content", "")
            m = re.search(r'\{[\s\S]*"text"\s*:\s*"[\s\S]*\}\s*\}\s*\]', reasoning)
            if m:
                content = m.group(0)

        if content.startswith("```"):
            content = re.sub(r'^```\w*\n?', '', content)
            content = re.sub(r'\n?```$', '', content)

        return json.loads(content.strip())
    except Exception as e:
        print(f"MiMo error: {e}")
        traceback.print_exc()
        return {"text": "", "errors": []}


def mimo_analyze_translation(image: Image.Image) -> dict:
    """分析中英文互译答卷，找出翻译错误"""
    img_b64 = image_to_base64(image)

    prompt = """这是一份英语词汇测试答卷，包含两种题型：
- 英译汉：给出英文单词，考生填写中文翻译
- 汉译英：给出中文词语，考生填写英文翻译

请逐题检查考生的翻译是否正确。

判断标准：
- 英译汉：考生写的中文是否准确对应英文单词的含义
- 汉译英：考生写的英文是否准确对应中文词语的含义
- 未作答的题目不需要标记
- 翻译完全正确的不需要标记
- 部分正确但有错误的，标记错误部分
- 拼写小瑕疵（如少写一个字母）如果导致意思不对就算错误，如果意思明确就不算

Output strict JSON only:
{"text":"全文OCR内容，每题一行，格式如：1. competent 有能力的","errors":[{"error":"考生填的错误答案","correction":"正确答案","category":"translation","message":"中文解释为什么错（如：approve意为批准，不是提高）","line":5,"x_pct":60}]}

Rules:
- "error" = 考生填写的错误内容（尽量简短）
- "correction" = 正确的翻译
- line = 1-based 行号（从答卷正文第一行开始数，跳过标题和姓名）
- x_pct = 错误内容在该行中的水平位置 (0=最左, 100=最右)
- 对于英译汉，x_pct 应指向考生写的中文答案的位置
- 对于汉译英，x_pct 应指向考生写的英文答案的位置
- 如果没有错误，返回 {"text":"...","errors":[]}
- Output ONLY JSON"""

    payload = {
        "model": MIMO_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": prompt}
        ]}],
        "max_tokens": 3000,
        "temperature": 0.1,
        "thinking": {"type": "disabled"},
    }

    try:
        resp = requests.post(MIMO_API_URL,
            headers={"Authorization": f"Bearer {MIMO_API_KEY}", "Content-Type": "application/json"},
            json=payload, timeout=120)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        if not content:
            reasoning = resp.json()["choices"][0]["message"].get("reasoning_content", "")
            m = re.search(r'\{[\s\S]*"text"\s*:\s*"[\s\S]*\}\s*\}\s*\]', reasoning)
            if m:
                content = m.group(0)

        if content.startswith("```"):
            content = re.sub(r'^```\w*\n?', '', content)
            content = re.sub(r'\n?```$', '', content)

        return json.loads(content.strip())
    except Exception as e:
        print(f"MiMo translation error: {e}")
        traceback.print_exc()
        return {"text": "", "errors": []}


# ========== 标注 ==========

def annotate_image(image: Image.Image, errors: list, text_lines: list, ocr_lines: list = None) -> Image.Image:
    if not errors or not text_lines:
        return image

    try:
        annotated = image.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated)
        font_label, font_small, font_tag = _load_fonts()
        width, height = annotated.size

        used_lines = {}  # 同一行多个错误时错开

        for error in errors:
            error_text = error.get("error", "")
            cat = error.get("category", "style")
            line_num = error.get("line", 0)
            x_pct = error.get("x_pct", 50)
            correction = error.get("correction", "")
            color = ERROR_COLORS.get(cat, (128, 128, 128))

            if line_num <= 0 or line_num > len(text_lines):
                print(f"  ✗ '{error_text}' -> line {line_num} out of range (max={len(text_lines)}), skip")
                continue

            # y 坐标：直接用检测到的文字行中心
            line_y = text_lines[line_num - 1]

            # x 坐标：优先用OCR文本计算字符偏移量（比MiMo x_pct精确得多）
            ocr_line = ocr_lines[line_num - 1] if ocr_lines and line_num <= len(ocr_lines) else ""
            ocr_x_pct, ocr_width_pct = compute_x_from_ocr(error_text, ocr_line)

            if ocr_x_pct is not None:
                # 用OCR字符偏移量（更精确）
                x_pct = ocr_x_pct
                print(f"  ✓ Using OCR-based x_pct={x_pct:.1f}% (error='{error_text}', line='{ocr_line[:50]}...')")
            else:
                # fallback到MiMo x_pct
                print(f"  ⚠ OCR match failed, fallback to MiMo x_pct={x_pct} (error='{error_text}')")

            # 用垂直投影检测该行实际文字边界
            text_left, text_right = detect_line_x_range(image, line_y)
            text_width = text_right - text_left
            print(f"  Line {line_num}: text range x={text_left}-{text_right} (width={text_width})")

            # 将 x_pct 映射到实际像素坐标，作为下划线的左端起点
            x_start = text_left + int(text_width * x_pct / 100)

            # 下划线宽度：基于错误词在行中的字符占比（更精确）
            if ocr_width_pct is not None and ocr_width_pct > 0:
                # 用OCR计算的宽度百分比
                underline_w = int(text_width * ocr_width_pct / 100)
            else:
                # fallback：按错误词字符数估算
                char_px = max(18, text_width // 35)
                underline_w = char_px * min(len(error_text.split()) * 2 + 1, 8)
            # 保证最小宽度和最大宽度
            min_w = int(text_width * 0.03)  # 至少3%行宽
            max_w = text_width // 2          # 不超过行宽一半
            underline_w = max(min_w, min(underline_w, max_w))
            x_end = min(x_start + underline_w, text_right + 10)

            # 同一行多个错误时上下错开
            offset = used_lines.get(line_num, 0)
            if offset > 0:
                line_y += offset * 18
            used_lines[line_num] = offset + 1

            y = line_y + 14  # 文字中心下方 14px

            # 3px 粗横线
            for dy in range(3):
                draw.line([(x_start, y + dy), (x_end, y + dy)], fill=color, width=1)

            # 左端小圆点
            draw.ellipse([x_start - 3, y - 1, x_start + 5, y + 5], fill=color)

            # 修正标签
            if correction:
                label = f"→ {correction[:20]}"
                lbox = draw.textbbox((0, 0), label, font=font_small)
                lw = lbox[2] - lbox[0]
                lh = lbox[3] - lbox[1]
                lx = x_end + 6
                ly = y - lh // 2
                if lx + lw > width - 10:
                    lx = x_start - lw - 6
                draw.rounded_rectangle(
                    [lx - 3, ly - 2, lx + lw + 3, ly + lh + 2],
                    radius=4, fill=(255, 255, 255), outline=color, width=1
                )
                draw.text((lx, ly), label, fill=color, font=font_small)

            print(f"  ✓ '{error_text}' -> line {line_num}, y={text_lines[line_num-1]}, y_underline={y}, x={x_start}-{x_end}")

        return annotated
    except Exception as e:
        print(f"annotate failed: {e}")
        traceback.print_exc()
        return image


# ========== API ==========

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
            image = resize_image(image)

            # Step 1: MiMo 分析（先获取行数，用于优化行检测）
            result = mimo_analyze_essay(image)
            ocr_text = result.get("text", "")
            errors = result.get("errors", [])
            print(f"MiMo found {len(errors)} errors")
            for e in errors:
                print(f"  - line {e.get('line')}: '{e.get('error')}' -> '{e.get('correction')}' x={e.get('x_pct')}")

            if not ocr_text.strip():
                results.append({"original_text": "", "errors": [],
                    "annotated_image": base64.b64encode(contents).decode("utf-8")})
                continue

            # Step 2: 根据 OCR 文本行数优化行检测
            ocr_lines_raw = [l for l in ocr_text.strip().split("\n") if l.strip()]
            ocr_line_count = len(ocr_lines_raw)
            text_lines = detect_text_lines(image, expected_count=ocr_line_count)
            print(f"Detected {len(text_lines)} essay lines (expected {ocr_line_count}): {text_lines[:15]}")

            all_errors.extend(errors)
            annotated = annotate_image(image, errors, text_lines, ocr_lines=ocr_lines_raw)
            buf = io.BytesIO()
            annotated.save(buf, format="JPEG", quality=85, optimize=True)

            results.append({
                "original_text": ocr_text, "errors": errors,
                "annotated_image": base64.b64encode(buf.getvalue()).decode("utf-8")
            })
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            results.append({"original_text": "", "errors": [],
                "annotated_image": base64.b64encode(contents).decode("utf-8")})

    categories = {}
    for error in all_errors:
        cat = error.get("category", "style")
        categories[cat] = categories.get(cat, 0) + 1

    return JSONResponse({"results": results,
        "summary": {"total_errors": len(all_errors), "categories": categories}})


@app.post("/api/check-translation")
async def check_translation(files: List[UploadFile] = File(...)):
    """中英文互译查错"""
    if len(files) > 2:
        raise HTTPException(status_code=400, detail="最多只能上传2张图片")

    results = []
    all_errors = []

    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            image = resize_image(image)

            result = mimo_analyze_translation(image)
            ocr_text = result.get("text", "")
            errors = result.get("errors", [])
            print(f"Translation check found {len(errors)} errors")
            for e in errors:
                print(f"  - line {e.get('line')}: '{e.get('error')}' -> '{e.get('correction')}' x={e.get('x_pct')}")

            if not ocr_text.strip():
                results.append({"original_text": "", "errors": [],
                    "annotated_image": base64.b64encode(contents).decode("utf-8")})
                continue

            ocr_lines_raw = [l for l in ocr_text.strip().split("\n") if l.strip()]
            ocr_line_count = len(ocr_lines_raw)
            text_lines = detect_text_lines(image, expected_count=ocr_line_count)
            print(f"Detected {len(text_lines)} lines (expected {ocr_line_count}): {text_lines[:15]}")

            all_errors.extend(errors)
            annotated = annotate_image(image, errors, text_lines, ocr_lines=ocr_lines_raw)
            buf = io.BytesIO()
            annotated.save(buf, format="JPEG", quality=85, optimize=True)

            results.append({
                "original_text": ocr_text, "errors": errors,
                "annotated_image": base64.b64encode(buf.getvalue()).decode("utf-8")
            })
        except Exception as e:
            print(f"Translation check error: {e}")
            traceback.print_exc()
            results.append({"original_text": "", "errors": [],
                "annotated_image": base64.b64encode(contents).decode("utf-8")})

    categories = {}
    for error in all_errors:
        cat = error.get("category", "translation")
        categories[cat] = categories.get(cat, 0) + 1

    return JSONResponse({"results": results,
        "summary": {"total_errors": len(all_errors), "categories": categories}})


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "v14-translation"}


@app.get("/")
async def root():
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Essay Grader API", "docs": "/docs"}


_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
