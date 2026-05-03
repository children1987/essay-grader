// ===== 调试工具 =====
const DEBUG_KEY = "essay_grader_debug";
let debugEnabled = localStorage.getItem(DEBUG_KEY) === "true";

function debugLog(type, msg, data) {
    const time = new Date().toLocaleTimeString();
    const entry = document.createElement("div");
    entry.className = "debug-entry debug-" + type;
    entry.innerHTML = `<span class="debug-time">${time}</span> <span class="debug-type">[${type.toUpperCase()}]</span> ${msg}`;
    if (data) {
        const pre = document.createElement("pre");
        pre.className = "debug-data";
        pre.textContent = typeof data === "string" ? data : JSON.stringify(data, null, 2);
        entry.appendChild(pre);
    }
    const log = document.getElementById("debug-log");
    if (log) {
        log.appendChild(entry);
        log.scrollTop = log.scrollHeight;
        const count = document.getElementById("debug-count");
        if (count) count.textContent = "(" + log.children.length + ")";
    }
    // 也输出到 console
    console[type === "error" ? "error" : "log"](`[EssayGrader] ${msg}`, data || "");
}

function debugClear() {
    const log = document.getElementById("debug-log");
    if (log) log.innerHTML = "";
    const count = document.getElementById("debug-count");
    if (count) count.textContent = "(0)";
}

// ===== API 地址（同源部署，自动适配） =====
const API_BASE = window.location.origin;
debugLog("info", "启动完成", {
    hostname: window.location.hostname,
    origin: window.location.origin,
    apiBase: API_BASE,
    userAgent: navigator.userAgent.substring(0, 80)
});

// ===== DOM 元素 =====
const homePage = document.getElementById("home-page");
const resultPage = document.getElementById("result-page");
const uploadArea = document.getElementById("upload-area");
const cameraInput = document.getElementById("camera-input");
const galleryInput = document.getElementById("gallery-input");
const cameraBtn = document.getElementById("camera-btn");
const galleryBtn = document.getElementById("gallery-btn");
const previewContainer = document.getElementById("preview-container");
const previewList = document.getElementById("preview-list");
const submitBtn = document.getElementById("submit-btn");
const backBtn = document.getElementById("back-btn");
const resultContainer = document.getElementById("result-container");
const resultList = document.getElementById("result-list");
const errorSummary = document.getElementById("error-summary");
const summaryContent = document.getElementById("summary-content");
const errorDetail = document.getElementById("error-detail");
const errorDetailList = document.getElementById("error-detail-list");

// ===== 调试面板折叠 =====
const debugToggle = document.getElementById("debug-toggle");
const debugLogEl = document.getElementById("debug-log");
debugToggle.addEventListener("click", () => {
    debugLogEl.style.display = debugLogEl.style.display === "none" ? "block" : "none";
});
debugLogEl.style.display = "none"; // 默认折叠

// ===== 状态 =====
let selectedFiles = [];

// ===== 事件绑定 =====
cameraBtn.addEventListener("click", () => cameraInput.click());
galleryBtn.addEventListener("click", () => galleryInput.click());
cameraInput.addEventListener("change", handleFileSelect);
galleryInput.addEventListener("change", handleFileSelect);
submitBtn.addEventListener("click", handleSubmit);
backBtn.addEventListener("click", goHome);

// ===== 健康检查 =====
(async function healthCheck() {
    try {
        debugLog("info", "健康检查: " + API_BASE + "/api/health");
        const resp = await fetch(API_BASE + "/api/health", { method: "GET" });
        debugLog("info", "健康检查响应", { status: resp.status, ok: resp.ok });
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        const data = await resp.json();
        debugLog("info", "服务正常", data);
    } catch (e) {
        debugLog("error", "健康检查失败: " + e.message);
    }
})();

// ===== 处理文件选择 =====
function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    debugLog("info", "选择文件: " + files.length + " 个", files.map(f => ({ name: f.name, size: f.size, type: f.type })));
    
    if (selectedFiles.length + files.length > 2) {
        alert("最多只能选择2张图片");
        return;
    }
    
    files.forEach(file => {
        if (file.type.startsWith("image/")) {
            selectedFiles.push(file);
        }
    });
    
    updatePreview();
    cameraInput.value = "";
    galleryInput.value = "";
}

// ===== 更新预览 =====
function updatePreview() {
    previewList.innerHTML = "";
    
    selectedFiles.forEach((file, index) => {
        const item = document.createElement("div");
        item.className = "preview-item";
        
        const img = document.createElement("img");
        img.src = URL.createObjectURL(file);
        
        const removeBtn = document.createElement("button");
        removeBtn.className = "remove-btn";
        removeBtn.textContent = "×";
        removeBtn.onclick = (e) => {
            e.stopPropagation();
            removeFile(index);
        };
        
        item.appendChild(img);
        item.appendChild(removeBtn);
        previewList.appendChild(item);
    });
    
    if (selectedFiles.length > 0) {
        previewContainer.classList.remove("hidden");
        uploadArea.style.display = "none";
        submitBtn.disabled = false;
    } else {
        previewContainer.classList.add("hidden");
        uploadArea.style.display = "block";
        submitBtn.disabled = true;
    }
}

// ===== 移除文件 =====
function removeFile(index) {
    debugLog("info", "移除文件 #" + index);
    selectedFiles.splice(index, 1);
    updatePreview();
}

// ===== 提交批改 =====
async function handleSubmit() {
    if (selectedFiles.length === 0) return;
    
    const btnText = submitBtn.querySelector(".btn-text");
    const btnLoading = submitBtn.querySelector(".btn-loading");
    btnText.classList.add("hidden");
    btnLoading.classList.remove("hidden");
    submitBtn.disabled = true;
    
    const url = API_BASE + "/api/grade";
    debugLog("info", "开始批改, 文件数: " + selectedFiles.length);
    debugLog("info", "请求地址: POST " + url);
    
    try {
        const formData = new FormData();
        selectedFiles.forEach((file, i) => {
            debugLog("info", "附件 #" + i + ": " + file.name + " (" + (file.size / 1024).toFixed(1) + " KB)");
            formData.append("files", file);
        });
        
        const startTime = Date.now();
        debugLog("info", "发送请求...");

        // 带超时和重试的 fetch
        let response = null;
        let lastError = null;
        const MAX_RETRIES = 1;
        const TIMEOUT_MS = 90000; // 90秒超时（Render Free 冷启动可能慢）

        for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
            try {
                if (attempt > 0) {
                    debugLog("info", "重试第 " + attempt + " 次...");
                    await new Promise(r => setTimeout(r, 3000)); // 等3秒再重试
                }
                const controller = new AbortController();
                const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
                response = await fetch(url, {
                    method: "POST",
                    body: formData,
                    signal: controller.signal
                });
                clearTimeout(timer);
                break; // 成功就跳出重试循环
            } catch (e) {
                lastError = e;
                debugLog("error", "请求失败 (attempt " + (attempt + 1) + "): " + e.message);
                if (attempt >= MAX_RETRIES) throw e;
            }
        }

        const elapsed = Date.now() - startTime;
        debugLog("info", "收到响应: HTTP " + response.status + " (" + elapsed + "ms)");

        if (!response.ok) {
            const errText = await response.text();
            debugLog("error", "服务器返回错误", { status: response.status, body: errText.substring(0, 500) });
            throw new Error("服务器错误 HTTP " + response.status + ": " + errText.substring(0, 100));
        }

        const result = await response.json();
        debugLog("info", "批改完成", {
            totalErrors: result.summary?.total_errors,
            categories: result.summary?.categories,
            resultsCount: result.results?.length
        });

        showResults(result);

    } catch (error) {
        debugLog("error", "批改失败: " + error.message, {
            name: error.name,
            stack: error.stack?.substring(0, 200)
        });
        alert("批改失败: " + error.message);
    } finally {
        btnText.classList.remove("hidden");
        btnLoading.classList.add("hidden");
        submitBtn.disabled = false;
    }
}

// ===== 显示结果 =====
function showResults(data) {
    resultList.innerHTML = "";
    
    if (data.results && data.results.length > 0) {
        data.results.forEach((item, i) => {
            debugLog("info", "结果 #" + i + ": " + item.errors.length + " 个错误, 文本长度 " + item.original_text.length);
            const resultItem = document.createElement("div");
            resultItem.className = "result-item";
            
            const img = document.createElement("img");
            img.src = "data:image/png;base64," + item.annotated_image;
            img.style.cursor = "pointer";
            img.addEventListener("click", () => openViewer(img.src));

            resultItem.appendChild(img);
            resultList.appendChild(resultItem);
        });
    }
    
    if (data.summary) {
        errorSummary.classList.remove("hidden");
        summaryContent.innerHTML = "";
        
        const categories = data.summary.categories || {};
        for (const [type, count] of Object.entries(categories)) {
            if (count > 0) {
                const tag = document.createElement("span");
                tag.className = "error-tag " + type;
                tag.textContent = getCategoryName(type) + ": " + count + "处";
                summaryContent.appendChild(tag);
            }
        }
        
        if (data.summary.total_errors === 0) {
            summaryContent.innerHTML = '<p style="color: #28a745; font-weight: 600;">🎉 未发现明显错误，作文很棒！</p>';
        }
        
        // 详细错误列表
        errorDetailList.innerHTML = "";
        if (data.results) {
            data.results.forEach(item => {
                if (item.errors && item.errors.length > 0) {
                    item.errors.forEach((err, i) => {
                        const div = document.createElement("div");
                        div.className = "error-item " + (err.category || "style");
                        const catName = getCategoryName(err.category || "style");
                        const lineNum = err.line ? ("第" + err.line + "行") : "";
                        div.innerHTML = '<span class="error-idx">#' + (i+1) + '</span>' +
                            '<span class="error-cat">' + catName + '</span>' +
                            (lineNum ? '<span class="error-line">' + lineNum + '</span>' : '') +
                            '<span class="error-old">' + (err.error || "") + '</span>' +
                            '<span class="error-arrow">→</span>' +
                            '<span class="error-new">' + (err.correction || "") + '</span>' +
                            (err.message ? '<div class="error-msg">' + err.message + '</div>' : '');
                        errorDetailList.appendChild(div);
                    });
                }
            });
        }
        if (data.summary && data.summary.total_errors > 0) {
            errorDetail.classList.remove("hidden");
        }
    }
    
    homePage.classList.remove("active");
    resultPage.classList.add("active");
}

function getCategoryName(type) {
    const names = { spelling: "拼写", grammar: "语法", punctuation: "标点", style: "表达" };
    return names[type] || type;
}

function goHome() {
    resultPage.classList.remove("active");
    homePage.classList.add("active");
    selectedFiles = [];
    updatePreview();
    resultList.innerHTML = "";
    errorSummary.classList.add("hidden");
    errorDetail.classList.add("hidden");
}

// ===== 全屏图片查看器 =====
const viewer = document.getElementById("image-viewer");
const viewerImg = document.getElementById("viewer-img");
const viewerContainer = document.getElementById("viewer-container");
const viewerClose = document.getElementById("viewer-close");

let scale = 1, panX = 0, panY = 0;
let lastDist = 0, lastMidX = 0, lastMidY = 0;
let isPanning = false, startPanX = 0, startPanY = 0;
let lastTap = 0;

function openViewer(src) {
    viewerImg.src = src;
    scale = 1; panX = 0; panY = 0;
    updateTransform();
    viewer.classList.remove("hidden");
    document.body.style.overflow = "hidden";
}

function closeViewer() {
    viewer.classList.add("hidden");
    viewerImg.src = "";
    document.body.style.overflow = "";
}

function updateTransform() {
    viewerImg.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
}

function clampPan() {
    if (scale <= 1) { panX = 0; panY = 0; return; }
    const rect = viewerContainer.getBoundingClientRect();
    const imgW = viewerImg.naturalWidth * scale;
    const imgH = viewerImg.naturalHeight * scale;
    const maxX = Math.max(0, (imgW - rect.width) / 2);
    const maxY = Math.max(0, (imgH - rect.height) / 2);
    panX = Math.max(-maxX, Math.min(maxX, panX));
    panY = Math.max(-maxY, Math.min(maxY, panY));
}

// 点击关闭按钮
viewerClose.addEventListener("click", closeViewer);

// 点击背景关闭（未缩放时）
viewerContainer.addEventListener("click", (e) => {
    if (e.target === viewerContainer && scale <= 1) closeViewer();
});

// 双击放大/还原
viewerContainer.addEventListener("click", (e) => {
    if (e.target === viewerImg || e.target === viewerContainer) {
        const now = Date.now();
        if (now - lastTap < 300) {
            if (scale > 1) {
                scale = 1; panX = 0; panY = 0;
            } else {
                scale = 2.5;
                const rect = viewerContainer.getBoundingClientRect();
                panX = (rect.width / 2 - e.clientX) * 1.5;
                panY = (rect.height / 2 - e.clientY) * 1.5;
            }
            clampPan();
            updateTransform();
            lastTap = 0;
        } else {
            lastTap = now;
        }
    }
});

// === 触摸事件：双指缩放 + 单指拖拽 ===
viewerContainer.addEventListener("touchstart", (e) => {
    if (e.touches.length === 2) {
        e.preventDefault();
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        lastDist = Math.hypot(dx, dy);
        lastMidX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
        lastMidY = (e.touches[0].clientY + e.touches[1].clientY) / 2;
    } else if (e.touches.length === 1 && scale > 1) {
        e.preventDefault();
        isPanning = true;
        startPanX = e.touches[0].clientX - panX;
        startPanY = e.touches[0].clientY - panY;
    }
}, { passive: false });

viewerContainer.addEventListener("touchmove", (e) => {
    if (e.touches.length === 2) {
        e.preventDefault();
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        const dist = Math.hypot(dx, dy);
        const midX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
        const midY = (e.touches[0].clientY + e.touches[1].clientY) / 2;

        // 缩放
        const ratio = dist / lastDist;
        scale = Math.max(0.5, Math.min(5, scale * ratio));
        lastDist = dist;

        // 平移（跟手中指）
        panX += midX - lastMidX;
        panY += midY - lastMidY;
        lastMidX = midX;
        lastMidY = midY;

        clampPan();
        updateTransform();
    } else if (e.touches.length === 1 && isPanning) {
        e.preventDefault();
        panX = e.touches[0].clientX - startPanX;
        panY = e.touches[0].clientY - startPanY;
        clampPan();
        updateTransform();
    }
}, { passive: false });

viewerContainer.addEventListener("touchend", (e) => {
    isPanning = false;
    if (e.touches.length < 2) lastDist = 0;
    // 如果缩放回到 1，重置位置
    if (scale < 0.8) { scale = 1; panX = 0; panY = 0; updateTransform(); }
});

// === 鼠标滚轮缩放（PC端） ===
viewerContainer.addEventListener("wheel", (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    scale = Math.max(0.5, Math.min(5, scale * delta));
    clampPan();
    updateTransform();
}, { passive: false });

// === 鼠标拖拽（PC端） ===
let mouseDown = false, mouseStartX = 0, mouseStartY = 0;
viewerContainer.addEventListener("mousedown", (e) => {
    if (scale > 1) {
        mouseDown = true;
        mouseStartX = e.clientX - panX;
        mouseStartY = e.clientY - panY;
        e.preventDefault();
    }
});
document.addEventListener("mousemove", (e) => {
    if (mouseDown) {
        panX = e.clientX - mouseStartX;
        panY = e.clientY - mouseStartY;
        clampPan();
        updateTransform();
    }
});
document.addEventListener("mouseup", () => { mouseDown = false; });
