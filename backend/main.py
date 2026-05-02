"""
英语作文批改 API 服务
使用小米 MiMo V2.5 omni 模型做手写 OCR + 语法检查
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

# MiMo API 配置
MIMO_API_URL = "https://token-plan-cn.xiaomimimo.com/v1/chat/completions"
MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "")
MIMO_MODEL = "mimo-v2-omni"

ERROR_COLORS = {
    "spelling": (255, 152, 0),    # 橙色
    "grammar": (211, 47, 47),     # 红色
    "punctuation": (2, 136, 209), # 蓝色
    "style": (56, 142, 60),       # 绿色
}

CATEGORY_NAMES = {
    "spelling": "拼写",
    "grammar": "语法",
    "punctuation": "标点",
    "style": "表达",
}


def resize_image(image: Image.Image, max_width: int = 1200) -> Image.Image:
    w, h = image.size
    if w <= max_width:
        return image
    new_h = int(h * max_width / w)
    return image.resize((max_width, new_h), Image.LANCZOS)


def image_to_base64(image: Image.Image, quality: int = 85) -> str:
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def mimo_ocr_and_check(image: Image.Image) -> dict:
    """
    用 MiMo V2.5 omni 模型同时完成 OCR + 语法检查
    """
    img_b64 = image_to_base64(image)

    prompt = """You are an English teacher grading a handwritten essay photo.

TASK:
1. Carefully read the English essay in this handwritten image
2. Transcribe the full text as accurately as possible
3. Find ALL grammar, spelling, punctuation, and expression errors
4. For EACH error line, estimate its vertical position in the image

OUTPUT FORMAT (strict JSON only, no markdown, no explanation):
{"text":"full transcribed essay text","errors":[{"error":"exact wrong text from essay","correction":"corrected text","category":"grammar|spelling|punctuation|style","message":"brief Chinese explanation","y_pct":35}]}

IMPORTANT RULES:
- "error" must be the EXACT wrong word/phrase from the essay
- "y_pct" = vertical position of that error LINE as percentage (0=top, 100=bottom). Look at where the text line sits on the paper. The title line is about 20-25%, first body line ~30-35%, each subsequent line adds ~5-8%.
- category must be one of: grammar, spelling, punctuation, style
- If no errors, return {"text":"...","errors":[]}
- Output ONLY the JSON object, nothing else"""

    payload = {
        "model": MIMO_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": prompt}
            ]
        }],
        "max_tokens": 8000,
        "temperature": 0.1,
    }

    try:
        resp = requests.post(
            MIMO_API_URL,
            headers={
                "Authorization": f"Bearer {MIMO_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        content = content.strip()
        if not content:
            reasoning = data["choices"][0]["message"].get("reasoning_content", "")
            json_match = re.search(r'\{[\s\S]*"text"\s*:\s*"[\s\S]*\}\s*\}\s*\]', reasoning)
            if json_match:
                content = json_match.group(0)
        if content.startswith("```"):
            content = re.sub(r'^```\w*\n?', '', content)
            content = re.sub(r'\n?```$', '', content)
        content = content.strip()

        result = json.loads(content)
        return result

    except json.JSONDecodeError as e:
        print(f"MiMo JSON parse error: {e}")
        print(f"Raw content: {content[:500]}")
        return {"text": "", "errors": []}
    except Exception as e:
        print(f"MiMo API error: {e}")
        traceback.print_exc()
        return {"text": "", "errors": []}


def _load_fonts():
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return (
                    ImageFont.truetype(p, 16),   # 标签字体
                    ImageFont.truetype(p, 13),   # 小字体
                    ImageFont.truetype(p, 11),   # 微字体
                    ImageFont.truetype(p, 14),   # 编号字体
                )
            except Exception:
                pass
    f = ImageFont.load_default()
    return f, f, f, f


def annotate_image(image: Image.Image, errors: list) -> Image.Image:
    """在原图上标注错误位置 - 专业版标注"""
    if not errors:
        return image
    try:
        return _annotate_pro(image, errors)
    except Exception as e:
        print(f"annotate failed: {e}")
        traceback.print_exc()
        return _annotate_fallback(image, errors)


def _annotate_pro(image: Image.Image, errors: list) -> Image.Image:
    """
    专业标注方案：
    - 左侧：彩色圆形编号标记（在页边空白处，不遮挡文字）
    - 中间：半透明细横线标记错误行位置
    - 右侧：修正建议标签（小字号，白色背景）
    - 同行错误：上下错开避免重叠
    """
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size
    font_label, font_small, font_tiny, font_num = _load_fonts()

    # 计算左右边距
    left_margin = 8
    right_margin = 8
    marker_x = left_margin + 14  # 编号圆点中心 x
    label_x_start = width - 220  # 右侧标签起始 x

    # 处理同行重叠：按 y_pct 排序，同一行的错开
    sorted_errors = sorted(enumerate(errors), key=lambda x: x[1].get("y_pct", 50))

    # 跟踪已占用的 y 区域，避免标签重叠
    used_y_zones = []

    for rank, (orig_idx, error) in enumerate(sorted_errors):
        y_pct = error.get("y_pct", 50)
        y_center = int(height * y_pct / 100)
        cat = error.get("category", "style")
        color = ERROR_COLORS.get(cat, (128, 128, 128))
        correction = error.get("correction", "")

        # 避免标签重叠：如果 y 区域已被占用，稍微偏移
        offset_y = 0
        for used_y in used_y_zones:
            if abs(y_center + offset_y - used_y) < 22:
                offset_y += 24
        used_y_zones.append(y_center + offset_y)
        y_adj = y_center + offset_y

        # 1. 左侧彩色圆形编号（在页边空白处）
        circle_r = 12
        cx, cy = marker_x, y_adj
        # 外圈（白色底）
        draw.ellipse([cx - circle_r - 2, cy - circle_r - 2,
                       cx + circle_r + 2, cy + circle_r + 2],
                      fill=(255, 255, 255), outline=color, width=2)
        # 内圈（彩色）
        draw.ellipse([cx - circle_r, cy - circle_r,
                       cx + circle_r, cy + circle_r],
                      fill=color)
        # 编号文字（白色）
        num_text = str(orig_idx + 1)
        bbox = draw.textbbox((0, 0), num_text, font=font_num)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((cx - tw // 2, cy - th // 2 - 1), num_text,
                   fill=(255, 255, 255), font=font_num)

        # 2. 从圆点到文字区域画一条细引导线（虚线效果）
        line_start_x = cx + circle_r + 4
        line_end_x = label_x_start - 10
        if line_end_x > line_start_x:
            dash_len = 6
            gap_len = 4
            x = line_start_x
            while x < line_end_x:
                x_end = min(x + dash_len, line_end_x)
                draw.line([(x, y_adj), (x_end, y_adj)], fill=color, width=1)
                x += dash_len + gap_len

        # 3. 右侧修正建议标签（白色圆角背景 + 彩色文字）
        if correction:
            label_text = f"→ {correction[:25]}"
            lbox = draw.textbbox((0, 0), label_text, font=font_small)
            lw = lbox[2] - lbox[0]
            lh = lbox[3] - lbox[1]
            pad_x, pad_y = 6, 3
            label_bg_x0 = label_x_start
            label_bg_y0 = y_adj - lh // 2 - pad_y
            label_bg_x1 = label_x_start + lw + pad_x * 2
            label_bg_y1 = y_adj + lh // 2 + pad_y

            # 白色背景 + 彩色边框
            draw.rounded_rectangle(
                [label_bg_x0, label_bg_y0, label_bg_x1, label_bg_y1],
                radius=6,
                fill=(255, 255, 255),
                outline=color,
                width=2
            )
            draw.text((label_bg_x0 + pad_x, label_bg_y0 + pad_y),
                       label_text, fill=color, font=font_small)

    return annotated


def _annotate_fallback(image: Image.Image, errors: list) -> Image.Image:
    """降级方案：底部显示错误列表"""
    annotated = image.copy()
    width, height = annotated.size
    font_label, font_small, _, _ = _load_fonts()
    error_height = min(len(errors) * 28 + 50, height // 3)
    new_height = height + error_height
    new_image = Image.new("RGB", (width, new_height), (255, 255, 255))
    new_image.paste(annotated, (0, 0))
    draw = ImageDraw.Draw(new_image)
    draw.rectangle([(0, height), (width, new_height)], fill=(33, 37, 41))
    draw.text((20, height + 12), f"发现 {len(errors)} 处错误:", fill=(255, 255, 255), font=font_label)
    y = height + 42
    for i, error in enumerate(errors[:10]):
        color = ERROR_COLORS.get(error.get("category", "style"), (128, 128, 128))
        cat_name = CATEGORY_NAMES.get(error.get("category", ""), "")
        line = f"{i+1}. [{cat_name}] {error.get('error', '')[:30]} → {error.get('correction', '')[:30]}"
        draw.text((20, y), line[:65], fill=color, font=font_small)
        y += 28
    return new_image


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

            result = mimo_ocr_and_check(image)
            ocr_text = result.get("text", "")
            errors = result.get("errors", [])

            if not ocr_text.strip():
                results.append({
                    "original_text": "",
                    "errors": [],
                    "annotated_image": base64.b64encode(contents).decode("utf-8")
                })
                continue

            all_errors.extend(errors)
            annotated = annotate_image(image, errors)
            buffer = io.BytesIO()
            annotated.save(buffer, format="PNG", optimize=True)
            annotated_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            results.append({
                "original_text": ocr_text,
                "errors": errors,
                "annotated_image": annotated_base64
            })
        except Exception as e:
            print(f"Error processing file: {e}")
            traceback.print_exc()
            results.append({
                "original_text": "",
                "errors": [],
                "annotated_image": base64.b64encode(contents).decode("utf-8")
            })

    categories = {}
    for error in all_errors:
        cat = error.get("category", "style")
        categories[cat] = categories.get(cat, 0) + 1
    return JSONResponse({
        "results": results,
        "summary": {"total_errors": len(all_errors), "categories": categories}
    })


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


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
