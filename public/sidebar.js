/**
 * FinDoc Agent — Right-side document panel.
 *
 * Injects a top-right toggle button into the Chainlit shell and a slide-in
 * sidebar listing the *user's* uploaded documents (system-default indexes
 * are excluded by the backend).
 *
 * Visual reference: claude.ai's right-side document/sources panel.
 *
 * Behaviour
 *   - Default closed; user opens manually via the toggle button.
 *   - Polls `/api/v1/documents` every 5s while open (so freshly-uploaded
 *     docs from the chat-side `+` button appear without manual reload).
 *   - Drag-and-drop / file picker upload directly inside the panel.
 *   - Reindex / delete actions per card.
 */

(function () {
  "use strict";

  // -------------------------------------------------------------------------
  // Config
  // -------------------------------------------------------------------------
  const cfgEl = document.getElementById("findoc-sidebar-config");
  const BACKEND_URL = (cfgEl && cfgEl.dataset.backendUrl) || "http://localhost:8001";
  const POLL_INTERVAL_MS = 5000;        // panel-open: full re-render
  const BG_POLL_INTERVAL_MS = 15000;    // panel-closed: badge-only refresh
  const ACCEPT_EXT = [".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif", ".webp"];

  // -------------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------------
  let toggleBtn = null;
  let toggleCountEl = null;
  let sidebarEl = null;
  let overlayEl = null;
  let bodyEl = null;
  let countEl = null;
  let dropZoneEl = null;
  let fileInputEl = null;
  let sidebarOpen = false;
  let pollTimer = null;
  let bgPollTimer = null;
  let docsCache = null;
  let initTimer = null;

  // -------------------------------------------------------------------------
  // SVG icons (zero deps)
  // -------------------------------------------------------------------------
  const ICONS = {
    doc: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>',
    close: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="18" height="18"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',
    refresh: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="15" height="15"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>',
    file: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><polyline points="13 2 13 9 20 9"/></svg>',
    upload: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
    pages: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="11" height="11"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
    clock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="11" height="11"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
    trash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="13" height="13"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2"/></svg>',
    reindex: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="13" height="13"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15A9 9 0 1 1 5.64 5.64L23 10"/></svg>',
  };

  // -------------------------------------------------------------------------
  // Utility
  // -------------------------------------------------------------------------
  function formatTs(ts) {
    if (!ts) return "—";
    try {
      const d = new Date(ts * 1000);
      const now = new Date();
      const diffDays = Math.floor((now - d) / 86400000);
      if (diffDays === 0) return "今天";
      if (diffDays === 1) return "昨天";
      if (diffDays < 7) return diffDays + " 天前";
      return d.toLocaleDateString("zh-CN", { month: "2-digit", day: "2-digit" });
    } catch (e) {
      return "—";
    }
  }

  function displayFilename(doc) {
    const raw = doc.source_filename || doc.doc_id || "";
    return raw.replace(/^[a-f0-9]{12}_/i, "");
  }

  function fileExtIcon(filename) {
    const ext = (filename.match(/\.[^.]+$/) || [""])[0].toLowerCase();
    return ext.replace(".", "").toUpperCase() || "DOC";
  }

  function buildPreviewFallback() {
    const fb = document.createElement("div");
    fb.className = "findoc-doc-preview-fallback";
    fb.innerHTML = '<span class="findoc-doc-preview-noimage">无预览</span>';
    return fb;
  }

  const STATUS_LABEL = { ready: "已就绪", encoding: "索引中", failed: "失败", queued: "排队中" };
  const STATUS_CLASS = { ready: "ready", encoding: "encoding", failed: "failed", queued: "queued" };

  // -------------------------------------------------------------------------
  // API
  // -------------------------------------------------------------------------
  async function apiCall(url, options) {
    const resp = await fetch(BACKEND_URL + url, options);
    if (!resp.ok) {
      let detail = "";
      try { detail = (await resp.json()).detail || resp.statusText; } catch (e) { /* */ }
      throw new Error(detail || ("HTTP " + resp.status));
    }
    return resp;
  }

  async function fetchDocs() {
    const resp = await apiCall("/api/v1/documents");
    return resp.json();
  }

  async function deleteDoc(docId) {
    await apiCall("/api/v1/documents/" + encodeURIComponent(docId), { method: "DELETE" });
  }

  async function reindexDoc(docId) {
    const resp = await apiCall("/api/v1/documents/" + encodeURIComponent(docId) + "/reindex", { method: "POST" });
    return resp.json();
  }

  function watchUpload(uploadId, onUpdate) {
    return new Promise(function (resolve, reject) {
      const source = new EventSource(BACKEND_URL + "/api/v1/upload/" + encodeURIComponent(uploadId) + "/status");
      source.onmessage = function (event) {
        let info = {};
        try { info = JSON.parse(event.data || "{}"); } catch (e) { return; }
        onUpdate(info);
        if (info.status === "done") { source.close(); resolve(info); }
        if (info.status === "failed") { source.close(); reject(new Error(info.message || "索引失败")); }
      };
      source.onerror = function () { source.close(); reject(new Error("无法读取索引进度")); };
    });
  }

  async function uploadFile(file, onProgress) {
    const fd = new FormData();
    fd.append("file", file, file.name);
    const resp = await fetch(BACKEND_URL + "/api/v1/upload", { method: "POST", body: fd });
    if (!resp.ok) {
      let detail = "";
      try { detail = (await resp.json()).detail || resp.statusText; } catch (e) { /* */ }
      throw new Error(detail || ("HTTP " + resp.status));
    }
    const result = await resp.json();
    await watchUpload(result.upload_id, onProgress);
    return result.doc_id;
  }

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  function renderSkeleton() {
    bodyEl.innerHTML = "";
    for (let i = 0; i < 3; i++) {
      const card = document.createElement("div");
      card.className = "findoc-doc-card";
      card.innerHTML =
        '<div class="findoc-skeleton" style="height:140px"></div>' +
        '<div style="padding:12px 14px">' +
        '<div class="findoc-skeleton" style="height:14px;width:70%;margin-bottom:8px"></div>' +
        '<div class="findoc-skeleton" style="height:11px;width:50%"></div>' +
        '</div>';
      bodyEl.appendChild(card);
    }
  }

  function renderEmpty() {
    bodyEl.innerHTML = "";
    const wrap = document.createElement("div");
    wrap.className = "findoc-sidebar-empty";
    wrap.innerHTML =
      '<div class="findoc-empty-icon">' + ICONS.file + '</div>' +
      '<p class="findoc-empty-title">暂无已上传文档</p>' +
      '<p class="findoc-empty-hint">将 PDF 或图片拖到上方<br/>或点击聊天框左侧 <b>+</b> 上传</p>';
    bodyEl.appendChild(wrap);
  }

  function renderDocCard(doc) {
    const card = document.createElement("div");
    card.className = "findoc-doc-card";
    card.setAttribute("data-doc-id", doc.doc_id);

    // Preview — backend inlines a base64 thumbnail in `doc.thumbnail` when an
    // image exists for this doc; otherwise we render a plain "no image" cell.
    // Either way, no follow-up HTTP request is needed.
    const preview = document.createElement("div");
    preview.className = "findoc-doc-preview";

    if (doc.thumbnail) {
      const img = document.createElement("img");
      img.src = doc.thumbnail;
      img.alt = "";
      img.decoding = "async";
      img.onerror = function () {
        img.remove();
        preview.appendChild(buildPreviewFallback());
      };
      preview.appendChild(img);
    } else {
      preview.appendChild(buildPreviewFallback());
    }

    // Status pill overlay
    const statusEl = document.createElement("span");
    statusEl.className = "findoc-doc-status " + (STATUS_CLASS[doc.status] || "");
    statusEl.textContent = STATUS_LABEL[doc.status] || doc.status;
    preview.appendChild(statusEl);

    card.appendChild(preview);

    // Body
    const cardBody = document.createElement("div");
    cardBody.className = "findoc-doc-body";

    const nameEl = document.createElement("div");
    nameEl.className = "findoc-doc-name";
    nameEl.textContent = displayFilename(doc);
    nameEl.title = doc.doc_id;
    cardBody.appendChild(nameEl);

    const meta = document.createElement("div");
    meta.className = "findoc-doc-meta";
    meta.innerHTML =
      "<span>" + ICONS.pages + " " + (doc.page_count || 0) + " 页</span>" +
      "<span>" + ICONS.clock + " " + formatTs(doc.created_at) + "</span>";
    cardBody.appendChild(meta);

    // Actions
    const actions = document.createElement("div");
    actions.className = "findoc-doc-actions";

    const reindexBtn = document.createElement("button");
    reindexBtn.className = "findoc-doc-action";
    reindexBtn.innerHTML = ICONS.reindex + '<span>重新索引</span>';
    reindexBtn.addEventListener("click", function (e) {
      e.stopPropagation();
      handleReindex(doc.doc_id, card);
    });
    actions.appendChild(reindexBtn);

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "findoc-doc-action danger";
    deleteBtn.innerHTML = ICONS.trash + '<span>删除</span>';
    deleteBtn.addEventListener("click", function (e) {
      e.stopPropagation();
      handleDelete(doc.doc_id, card);
    });
    actions.appendChild(deleteBtn);

    cardBody.appendChild(actions);
    card.appendChild(cardBody);
    return card;
  }

  function diffDocs(prev, next) {
    if (!prev || prev.length !== next.length) return true;
    for (let i = 0; i < next.length; i++) {
      const a = prev[i] || {};
      const b = next[i] || {};
      if (a.doc_id !== b.doc_id || a.status !== b.status || a.page_count !== b.page_count) return true;
    }
    return false;
  }

  async function renderDocs(showSkeleton) {
    if (showSkeleton) renderSkeleton();
    let docs;
    try {
      docs = await fetchDocs();
    } catch (e) {
      bodyEl.innerHTML = "";
      const err = document.createElement("div");
      err.className = "findoc-sidebar-empty";
      err.innerHTML = '<p class="findoc-empty-title" style="color:#dc2626">加载失败</p>' +
        '<p class="findoc-empty-hint">' + (e.message || "请检查后端是否启动") + '</p>';
      bodyEl.appendChild(err);
      updateToggleBadge(0);
      return;
    }

    const changed = diffDocs(docsCache, docs);
    docsCache = docs;
    updateToggleBadge(docs.length);

    if (!changed && bodyEl.querySelector(".findoc-doc-card")) return;

    bodyEl.innerHTML = "";
    if (!docs || docs.length === 0) {
      renderEmpty();
      return;
    }
    docs.forEach(function (doc) { bodyEl.appendChild(renderDocCard(doc)); });
  }

  // -------------------------------------------------------------------------
  // Actions
  // -------------------------------------------------------------------------
  async function handleDelete(docId, cardEl) {
    if (!confirm('确定删除文档 "' + displayFilename({ doc_id: docId }) + '" 吗？此操作不可撤销。')) return;
    cardEl.style.opacity = "0.5";
    cardEl.style.pointerEvents = "none";
    try {
      await deleteDoc(docId);
      cardEl.remove();
      docsCache = (docsCache || []).filter(function (d) { return d.doc_id !== docId; });
      if (docsCache.length === 0) renderEmpty();
      updateToggleBadge(docsCache.length);
    } catch (e) {
      cardEl.style.opacity = "1";
      cardEl.style.pointerEvents = "auto";
      alert("删除失败: " + e.message);
    }
  }

  async function handleReindex(docId, cardEl) {
    const status = cardEl.querySelector(".findoc-doc-status");
    if (status) {
      status.textContent = "索引中...";
      status.className = "findoc-doc-status encoding";
    }
    try {
      const result = await reindexDoc(docId);
      await watchUpload(result.upload_id, function (info) {
        if (status) status.textContent = info.status === "done" ? "已就绪" : (info.message || "索引中...");
      });
      renderDocs(false);
    } catch (e) {
      if (status) {
        status.textContent = "失败";
        status.className = "findoc-doc-status failed";
      }
      alert("重新索引失败: " + e.message);
    }
  }

  async function handleUploadFiles(files) {
    if (!files || files.length === 0) return;
    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      const ext = ("." + (f.name.split(".").pop() || "")).toLowerCase();
      if (ACCEPT_EXT.indexOf(ext) === -1) {
        alert("跳过不支持的文件: " + f.name + "（仅支持 PDF / 图片）");
        continue;
      }

      const placeholder = document.createElement("div");
      placeholder.className = "findoc-doc-card uploading";
      placeholder.innerHTML =
        '<div class="findoc-doc-preview"><div class="findoc-doc-preview-fallback">' +
        '<div class="findoc-doc-preview-icon">' + ICONS.upload + '</div>' +
        '<span class="findoc-doc-preview-ext">' + fileExtIcon(f.name) + '</span></div></div>' +
        '<div class="findoc-doc-body">' +
        '<div class="findoc-doc-name">' + f.name + '</div>' +
        '<div class="findoc-doc-progress"><div class="findoc-doc-progress-bar"></div></div>' +
        '<div class="findoc-doc-progress-msg">准备上传…</div>' +
        '</div>';

      // Hide empty state if it's showing
      const empty = bodyEl.querySelector(".findoc-sidebar-empty");
      if (empty) empty.remove();
      bodyEl.insertBefore(placeholder, bodyEl.firstChild);

      const bar = placeholder.querySelector(".findoc-doc-progress-bar");
      const msgEl = placeholder.querySelector(".findoc-doc-progress-msg");

      try {
        await uploadFile(f, function (info) {
          if (bar) bar.style.width = Math.max(5, Math.round((info.pct || 0) * 100)) + "%";
          if (msgEl) msgEl.textContent = info.message || info.status || "处理中…";
        });
        placeholder.remove();
        renderDocs(false);
      } catch (e) {
        placeholder.classList.add("failed");
        if (msgEl) msgEl.textContent = "失败: " + e.message;
        if (bar) bar.style.background = "#dc2626";
        setTimeout(function () { placeholder.remove(); renderDocs(false); }, 4000);
      }
    }
  }

  // -------------------------------------------------------------------------
  // Toggle badge
  // -------------------------------------------------------------------------
  function updateToggleBadge(count) {
    if (!toggleBtn) return;
    if (toggleCountEl) toggleCountEl.textContent = String(count || 0);
    if (countEl) countEl.textContent = String(count || 0);
    toggleBtn.classList.toggle("has-docs", count > 0);
  }

  // -------------------------------------------------------------------------
  // Open / close + polling
  // -------------------------------------------------------------------------
  function startPolling() {
    stopPolling();
    pollTimer = window.setInterval(function () {
      if (sidebarOpen) renderDocs(false);
    }, POLL_INTERVAL_MS);
  }

  function stopPolling() {
    if (pollTimer) {
      window.clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  // Background-poll the count only, so the toggle pill reflects uploads done
  // through the chat-side `+` button even before the user opens the panel.
  function startBgPoll() {
    if (bgPollTimer) return;
    bgPollTimer = window.setInterval(function () {
      if (sidebarOpen) return;  // open-poll path handles this
      fetchDocs().then(function (docs) {
        const prevLen = docsCache ? docsCache.length : 0;
        docsCache = docs;
        if ((docs ? docs.length : 0) !== prevLen) {
          updateToggleBadge(docs ? docs.length : 0);
        }
      }).catch(function () { /* silent */ });
    }, BG_POLL_INTERVAL_MS);
  }

  function openSidebar() {
    if (sidebarOpen) return;
    sidebarOpen = true;
    sidebarEl.classList.add("open");
    overlayEl.classList.add("open");
    renderDocs(true);
    startPolling();
  }

  function closeSidebar() {
    if (!sidebarOpen) return;
    sidebarOpen = false;
    sidebarEl.classList.remove("open");
    overlayEl.classList.remove("open");
    stopPolling();
  }

  function toggleSidebar() {
    if (sidebarOpen) closeSidebar();
    else openSidebar();
  }

  // -------------------------------------------------------------------------
  // DOM construction
  // -------------------------------------------------------------------------
  function createSidebar() {
    if (!document.body) return false;

    // Overlay
    overlayEl = document.createElement("div");
    overlayEl.className = "findoc-sidebar-overlay";
    overlayEl.addEventListener("click", closeSidebar);
    document.body.appendChild(overlayEl);

    // Panel
    sidebarEl = document.createElement("aside");
    sidebarEl.className = "findoc-sidebar";

    // Header
    const header = document.createElement("div");
    header.className = "findoc-sidebar-header";

    const titleWrap = document.createElement("div");
    titleWrap.className = "findoc-sidebar-title";
    titleWrap.innerHTML = '<span class="findoc-sidebar-title-icon">' + ICONS.doc + '</span>' +
      '<span class="findoc-sidebar-title-text">文档库</span>';
    countEl = document.createElement("span");
    countEl.className = "findoc-sidebar-count";
    countEl.textContent = "0";
    titleWrap.appendChild(countEl);
    header.appendChild(titleWrap);

    const headerActions = document.createElement("div");
    headerActions.className = "findoc-sidebar-header-actions";

    const refreshBtn = document.createElement("button");
    refreshBtn.className = "findoc-sidebar-iconbtn";
    refreshBtn.innerHTML = ICONS.refresh;
    refreshBtn.title = "刷新";
    refreshBtn.setAttribute("aria-label", "刷新");
    refreshBtn.addEventListener("click", function () {
      refreshBtn.classList.add("spinning");
      renderDocs(false).finally(function () {
        setTimeout(function () { refreshBtn.classList.remove("spinning"); }, 400);
      });
    });
    headerActions.appendChild(refreshBtn);

    const closeBtn = document.createElement("button");
    closeBtn.className = "findoc-sidebar-iconbtn";
    closeBtn.innerHTML = ICONS.close;
    closeBtn.title = "关闭";
    closeBtn.setAttribute("aria-label", "关闭");
    closeBtn.addEventListener("click", closeSidebar);
    headerActions.appendChild(closeBtn);

    header.appendChild(headerActions);
    sidebarEl.appendChild(header);

    // Drag-drop / picker
    dropZoneEl = document.createElement("div");
    dropZoneEl.className = "findoc-sidebar-dropzone";
    dropZoneEl.innerHTML =
      '<div class="findoc-dropzone-icon">' + ICONS.upload + '</div>' +
      '<div class="findoc-dropzone-text">拖入文件或 <span class="findoc-dropzone-link">点击上传</span></div>' +
      '<div class="findoc-dropzone-hint">PDF · PNG · JPG · 最大 100MB</div>';

    fileInputEl = document.createElement("input");
    fileInputEl.type = "file";
    fileInputEl.accept = ACCEPT_EXT.join(",");
    fileInputEl.multiple = true;
    fileInputEl.style.display = "none";
    fileInputEl.addEventListener("change", function () {
      handleUploadFiles(fileInputEl.files);
      fileInputEl.value = "";
    });
    dropZoneEl.appendChild(fileInputEl);

    dropZoneEl.addEventListener("click", function () { fileInputEl.click(); });
    dropZoneEl.addEventListener("dragover", function (e) {
      e.preventDefault();
      dropZoneEl.classList.add("dragover");
    });
    dropZoneEl.addEventListener("dragleave", function () { dropZoneEl.classList.remove("dragover"); });
    dropZoneEl.addEventListener("drop", function (e) {
      e.preventDefault();
      dropZoneEl.classList.remove("dragover");
      handleUploadFiles(e.dataTransfer.files);
    });
    sidebarEl.appendChild(dropZoneEl);

    // Body
    bodyEl = document.createElement("div");
    bodyEl.className = "findoc-sidebar-body";
    sidebarEl.appendChild(bodyEl);

    document.body.appendChild(sidebarEl);

    // Toggle pill button
    toggleBtn = document.createElement("button");
    toggleBtn.className = "findoc-sidebar-toggle";
    toggleBtn.type = "button";
    toggleBtn.title = "我的文档";
    toggleBtn.setAttribute("aria-label", "我的文档");
    toggleBtn.innerHTML =
      '<span class="findoc-toggle-icon">' + ICONS.doc + '</span>' +
      '<span class="findoc-toggle-label">文档</span>' +
      '<span class="findoc-toggle-count">0</span>';
    toggleCountEl = toggleBtn.querySelector(".findoc-toggle-count");
    toggleBtn.addEventListener("click", toggleSidebar);
    document.body.appendChild(toggleBtn);

    // Esc to close
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && sidebarOpen) closeSidebar();
    });

    // Listen for upload notifications from chainlit-side uploads
    window.addEventListener("findoc-docs-changed", function () {
      if (sidebarOpen) renderDocs(false);
      else fetchDocs().then(function (docs) {
        docsCache = docs;
        updateToggleBadge(docs ? docs.length : 0);
      }).catch(function () { /* silent */ });
    });

    return true;
  }

  // -------------------------------------------------------------------------
  // Init
  // -------------------------------------------------------------------------
  function init() {
    if (document.querySelector(".findoc-sidebar-toggle")) return;
    if (!createSidebar()) return;

    // Background fetch so the toggle badge reflects reality before user opens
    fetchDocs().then(function (docs) {
      docsCache = docs;
      updateToggleBadge(docs ? docs.length : 0);
    }).catch(function () { /* silent */ });

    startBgPoll();
  }

  function scheduleInit() {
    window.clearTimeout(initTimer);
    initTimer = window.setTimeout(init, 200);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", scheduleInit);
  } else {
    scheduleInit();
  }

  // Re-inject if Chainlit's React shell wipes the DOM
  new MutationObserver(function () {
    if (!document.querySelector(".findoc-sidebar-toggle")) scheduleInit();
  }).observe(document.documentElement, { childList: true, subtree: true });
})();
