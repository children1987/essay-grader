"""
英语作文批改 API 服务 (v2 - 百度手写OCR + 字符级坐标标注)
"""
import os
import io
import json
import time
import base64
import urllib.request
import urllib.parse
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from PIL import Image, ImageDraw, ImageFont

# 初始化应用
app = FastAPI(title="英语作文批改 API")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 百度 OCR 配置
BAIDU_API_KEY = os.environ.get("BAIDU_API_KEY", "")
BAIDU_SECRET_KEY = os.environ.get("BAIDU_SECRET_KEY", "")
_baidu_token = {"access_token": None, "expires_at": 0}

# 错误类型颜色配置
ERROR_COLORS = {
    'spelling': (255, 140, 0),     # 橙色
    'grammar': (220, 40, 40),      # 红色
    'punctuation': (0, 120, 200),  # 蓝色
    'style': (40, 160, 60),        # 绿色
}

CATEGORY_NAMES = {
    'spelling': '拼写',
    'grammar': '语法',
    'punctuation': '标点',
    'style': '表达',
}


# ====== 百度 OCR 客户端 ======

def get_baidu_access_token() -> str:
    """获取百度 Access Token（缓存30天）"""
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
        raise Exception(f"百度 token 获取失败: {data}")

    _baidu_token["access_token"] = data["access_token"]
    _baidu_token["expires_at"] = now + data.get("expires_in", 2592000) - 60
    return _baidu_token["access_token"]


def baidu_handwriting_ocr(image_bytes: bytes) -> dict:
    """
    调用百度手写文字识别 API
    返回: {"lines": [{"text": str, "bbox": {left,top,width,height}, "chars": [{"char": str, "bbox": {...}}]}]}
    """
    token = get_baidu_access_token()
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    url = (
        f"https://aip.baidubce.com/rest/2.0/ocr/v1/handwriting"
        f"?access_token={token}"
    )
    body = urllib.parse.urlencode({
        "image": img_b64,
        "recognize_granularity": "small",    # 字符级坐标
        "eng_granularity": "word",           # 英文按单词
        "language_type": "CHN_ENG",          # 中英文
        "probability": "true",
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    if "error_code" in data and data["error_code"] != 0:
        raise Exception(f"百度 OCR 错误: {data.get('error_msg', data)}")

    # 整理返回格式
    lines = []
    for item in data.get("words_result", []):
        chars = []
        for c in item.get("chars", []):
            chars.append({
                "char": c["char"],
                "bbox": c["location"]  # {left, top, width, height}
            })
        lines.append({
            "text": item["words"],
            "bbox": item["location"],  # {left, top, width, height}
            "chars": chars,
        })

    return {"lines": lines}


# ====== 语法检查（LanguageTool） ======

def categorize_error(rule_id: str, message: str) -> str:
    """根据规则ID和消息分类错误类型"""
    rule_lower = rule_id.lower()
    msg_lower = message.lower()

    if 'spell' in rule_lower or 'misspell' in rule_lower:
        return 'spelling'
    if 'punct' in rule_lower or 'comma' in rule_lower or 'period' in rule_lower:
        return 'punctuation'
    if any(x in msg_lower for x in ['punctuation', 'comma', 'apostrophe']):
        return 'punctuation'
    if any(x in rule_lower for x in ['grammar', 'subject', 'verb', 'tense', 'agreement']):
        return 'grammar'
    if any(x in msg_lower for x in ['verb', 'tense', 'subject', 'agreement', 'article']):
        return 'grammar'
    return 'style'


def check_grammar(text: str) -> list:
    """检查语法错误（延迟加载 LanguageTool）"""
    from language_tool_python import LanguageTool
    tool = LanguageTool('en-US')
    matches = tool.check(text)
    errors = []

    for match in matches:
        errors.append({
            'message': match.message,
            'replacements': match.replacements[:3],
            'offset': match.offset,
            'length': match.errorLength,
            'context': text[max(0, match.offset-20):match.offset+match.errorLength+20],
            'category': categorize_error(match.ruleId, match.message),
            'rule_id': match.ruleId,
        })
    return errors


# ====== 坐标映射 ======

def build_char_map(ocr_lines: list) -> tuple:
    """
    从 OCR 行数据构建字符到坐标的映射
    返回: (full_text, char_coords)
    char_coords[i] = {"char": str, "bbox": {left,top,width,height}, "line_idx": int}
    """
    full_text = ""
    char_coords = []

    for line_idx, line in enumerate(ocr_lines):
        for ch_data in line["chars"]:
            char_coords.append({
                "char": ch_data["char"],
                "bbox": ch_data["bbox"],
                "line_idx": line_idx,
            })
            full_text += ch_data["char"]

        # 行之间加换行符（OCR 不返回换行，需要手动加）
        if line_idx < len(ocr_lines) - 1:
            char_coords.append({
                "char": "\n",
                "bbox": None,
                "line_idx": line_idx,
            })
            full_text += "\n"

    return full_text, char_coords


def map_errors_to_coords(errors: list, full_text: str, char_coords: list) -> list:
    """
    将 LanguageTool 错误映射到 OCR 字符坐标
    LanguageTool offset 是基于 OCR 文本的字符偏移
    """
    mapped = []
    for error in errors:
        offset = error["offset"]
        length = error["length"]

        # 收集错误范围内的字符坐标
        char_bboxes = []
        for i in range(offset, min(offset + length, len(char_coords))):
            cc = char_coords[i]
            if cc["bbox"] is not None:
                char_bboxes.append(cc["bbox"])

        if not char_bboxes:
            continue

        # 计算错误词的整体 bbox
        left = min(b["left"] for b in char_bboxes)
        top = min(b["top"] for b in char_bboxes)
        right = max(b["left"] + b["width"] for b in char_bboxes)
        bottom = max(b["top"] + b["height"] for b in char_bboxes)

        # 获取错误文本
        error_text = full_text[offset:offset+length].strip()

        mapped.append({
            **error,
            "error_text": error_text,
            "word_bbox": {
                "left": left,
                "top": top,
                "width": right - left,
                "height": bottom - top,
            }
        })

    return mapped


# ====== 图片标注 ======

def annotate_image(image: Image.Image, mapped_errors: list) -> Image.Image:
    """在图片上用下划线标注错误位置"""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    # 加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    for error in mapped_errors:
        bbox = error["word_bbox"]
        color = ERROR_COLORS.get(error["category"], (128, 128, 128))
        category = CATEGORY_NAMES.get(error["category"], "?")

        x = bbox["left"]
        y = bbox["top"]
        w = bbox["width"]
        h = bbox["height"]

        # 画下划线（粗线，错误词下方 2px）
        line_y = y + h + 2
        draw.line([(x, line_y), (x + w, line_y)], fill=color, width=3)

        # 在下划线下面画修正建议
        correction = error.get("replacements", [""])[0] if error.get("replacements") else ""
        if correction:
            label = f"→ {correction}"
            draw.text((x, line_y + 3), label, fill=color, font=small_font)

    return annotated


# ====== API 端点 ======

@app.post("/api/grade")
async def grade_essay(files: List[UploadFile] = File(...)):
    """批改英语作文"""
    if len(files) > 2:
        raise HTTPException(status_code=400, detail="最多只能上传2张图片")

    results = []
    all_errors = []

    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # 1. 百度 OCR 提取文字 + 字符坐标
        try:
            ocr_result = baidu_handwriting_ocr(contents)
        except Exception as e:
            results.append({
                'original_text': f'[OCR 失败: {str(e)}]',
                'errors': [],
                'annotated_image': base64.b64encode(contents).decode('utf-8'),
            })
            continue

        ocr_lines = ocr_result["lines"]
        if not ocr_lines:
            results.append({
                'original_text': '',
                'errors': [],
                'annotated_image': base64.b64encode(contents).decode('utf-8'),
            })
            continue

        # 2. 构建全文 + 字符坐标映射
        full_text, char_coords = build_char_map(ocr_lines)

        if not full_text.strip():
            results.append({
                'original_text': '',
                'errors': [],
                'annotated_image': base64.b64encode(contents).decode('utf-8'),
            })
            continue

        # 3. 语法检查
        errors = check_grammar(full_text)
        all_errors.extend(errors)

        # 4. 映射错误到坐标
        mapped_errors = map_errors_to_coords(errors, full_text, char_coords)

        # 5. 标注图片
        annotated = annotate_image(image, mapped_errors)

        # 6. 输出
        buffer = io.BytesIO()
        annotated.save(buffer, format='JPEG', quality=85)
        annotated_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        results.append({
            'original_text': full_text,
            'errors': [
                {
                    'error': e.get('error_text', ''),
                    'message': e['message'],
                    'replacements': e.get('replacements', []),
                    'category': e['category'],
                    'rule_id': e.get('rule_id', ''),
                }
                for e in mapped_errors
            ],
            'annotated_image': annotated_base64,
        })

    # 统计
    categories = {}
    for error in all_errors:
        cat = error['category']
        categories[cat] = categories.get(cat, 0) + 1

    return JSONResponse({
        'results': results,
        'summary': {
            'total_errors': len(all_errors),
            'categories': categories,
        }
    })


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "version": "v2-baidu-ocr"}


# 挂载静态文件（前端）
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
