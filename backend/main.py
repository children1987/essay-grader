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

def detect_text_lines(image: Image.Image) -> list:
    """水平投影分析检测文字行中心 y 坐标"""
    gray = image.convert("L")
    w, h = gray.size
    pixels = gray.load()

    threshold = 140
    projection = []
    for y in range(h):
        count = sum(1 for x in range(0, w, 3) if pixels[x, y] < threshold)
        projection.append(count)

    # Moving average
    window = 7
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

    return peaks


def detect_line_x_range(image: Image.Image, y_center: int) -> tuple:
    """水平投影检测某行文字的左右边界 (x_start, x_end)"""
    gray = image.convert("L")
    w, h = gray.size
    pixels = gray.load()
    threshold = 140
    band = 12  # 上下扫描范围

    # 垂直投影：每列有多少暗像素
    projection = []
    for x in range(w):
        count = sum(1 for y in range(max(0, y_center - band), min(h, y_center + band))
                    if pixels[x, y] < threshold)
        projection.append(count)

    # 找到有文字的列范围（投影 > 0 的区域）
    avg = sum(projection) / len(projection) if projection else 0
    text_cols = [i for i, v in enumerate(projection) if v > avg * 0.1]

    if len(text_cols) < 5:
        return (int(w * 0.05), int(w * 0.95))

    return (text_cols[0], text_cols[-1])


def y_pct_to_line(y_pct: float, text_lines: list, img_height: int) -> int:
    """将 y 百分比转换为最近的文字行索引"""
    y = img_height * y_pct / 100
    best_idx = 0
    best_dist = float('inf')
    for i, ly in enumerate(text_lines):
        dist = abs(y - ly)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


# ========== MiMo ==========

def mimo_analyze_essay(image: Image.Image) -> dict:
    img_b64 = image_to_base64(image)

    prompt = """Analyze this English essay image. Find ALL grammar/spelling/punctuation errors.

Output strict JSON only:
{"text":"full text","errors":[{"error":"wrong text","correction":"correct","category":"grammar|spelling|punctuation|style","message":"Chinese explanation","line":3,"x_pct":30}]}

Rules:
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


# ========== 标注 ==========

def annotate_image(image: Image.Image, errors: list, text_lines: list) -> Image.Image:
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

            # x 坐标：用水平投影检测该行实际文字边界
            text_left, text_right = detect_line_x_range(image, line_y)
            text_width = text_right - text_left
            print(f"  Line {line_num}: text range x={text_left}-{text_right} (width={text_width})")

            # x_pct = 错误词在整行中的起始百分比（0=行首，100=行尾）
            # 将 x_pct 映射到实际像素坐标，作为下划线的左端起点
            x_start = text_left + int(text_width * x_pct / 100)

            # 下划线宽度：按错误词字符数估算（手写体约 18-22px/字符）
            char_px = max(18, text_width // 35)
            underline_w = max(len(error_text) * char_px, 80)
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

            # Step 1: 检测文字行
            text_lines = detect_text_lines(image)
            print(f"Detected {len(text_lines)} text lines: {text_lines[:15]}")

            # Step 2: MiMo 分析
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

            all_errors.extend(errors)
            annotated = annotate_image(image, errors, text_lines)
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


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "v9-x-fix"}


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
