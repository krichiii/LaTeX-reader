// LaTeX Reader frontend. Plain script (not a module) so index.html also works when
// opened straight from disk via file://.
(() => {
  "use strict";

  const params = new URLSearchParams(location.search);
  const API_BASE = (params.get("api") || window.LATEX_OCR_API || "http://127.0.0.1:5000").replace(/\/+$/, "");
  const KATEX_JS = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js";
  // From the backend README: on the demo images every reading below 0.75 was wrong.
  const LOW_CONFIDENCE = 0.75;
  const IMAGE_EXTENSION = /\.(png|jpe?g|gif|bmp|webp|tiff?)$/i;
  const SERVER_RETRY_MS = 5000;

  const $ = (id) => document.getElementById(id);
  const els = {
    dropzone: $("dropzone"),
    fileInput: $("file-input"),
    fileList: $("file-list"),
    fileCount: $("file-count"),
    convertBtn: $("convert-btn"),
    clearBtn: $("clear-btn"),
    output: $("output"),
    preview: $("preview"),
    copyBtn: $("copy-btn"),
    copyLabel: $("copy-label"),
    copyIcon: $("copy-icon"),
    formatSelect: $("format-select"),
    viewTabs: $("view-tabs"),
    progress: $("progress"),
    empty: $("empty"),
    specimen: $("specimen"),
    resultMeta: $("result-meta"),
    server: $("server"),
    serverText: $("server-text"),
    toast: $("toast"),
    themeBtn: $("theme-btn"),
    themeColor: $("theme-color"),
  };

  /**
   * Each item: { id, file, url, status, result, error, el }
   * status: "pending" -> "queued" -> "working" -> "done" | "error"
   */
  const state = {
    items: [],
    nextId: 1,
    converting: false,
    runDone: 0, // images finished in the current run, for the progress bar
    controller: null,
    view: loadSetting("view", ["text", "preview"]),
    format: loadSetting("format", ["latex", "markdown"]),
    katex: "loading", // "loading" | "ready" | "failed"
  };

  class NetworkError extends Error {}

  // ---------- Settings (per browser, optional) ----------

  function loadSetting(key, allowed) {
    try {
      const value = localStorage.getItem(`latex-reader.${key}`);
      if (allowed.includes(value)) return value;
    } catch { /* storage blocked */ }
    return allowed[0];
  }

  function saveSetting(key, value) {
    try { localStorage.setItem(`latex-reader.${key}`, value); } catch { /* storage blocked */ }
  }

  // ---------- Color scheme ----------

  // The inline script in index.html already set data-theme before the first paint.
  const systemDark = matchMedia("(prefers-color-scheme: dark)");

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    els.themeColor.content = theme === "dark" ? "#111413" : "#edeeec";
    els.themeBtn.toggleAttribute("selected", theme === "dark");
  }

  function bindTheme() {
    applyTheme(document.documentElement.dataset.theme === "dark" ? "dark" : "light");

    els.themeBtn.addEventListener("change", () => {
      const theme = els.themeBtn.selected ? "dark" : "light";
      saveSetting("theme", theme);
      applyTheme(theme);
    });

    // Until the button is used, follow the system setting when it changes.
    systemDark.addEventListener("change", (event) => {
      if (loadSetting("theme", ["", "light", "dark"])) return;
      applyTheme(event.matches ? "dark" : "light");
    });
  }

  // ---------- Helpers ----------

  function plural(count, word) {
    return `${count} ${word}${count === 1 ? "" : "s"}`;
  }

  function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  // Material Symbols paths, inlined so icons need no icon font.
  const ICONS = {
    close: "M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z",
    copy: "M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z",
    check: "M9 16.17 4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z",
    warning: "M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z",
    error: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z",
  };

  function icon(name, slot) {
    const node = el("md-icon");
    if (slot) node.slot = slot;
    node.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="${ICONS[name]}"/></svg>`;
    return node;
  }

  let toastTimer = 0;
  function showToast(message, tone = "info") {
    els.toast.textContent = message;
    els.toast.dataset.tone = tone;
    els.toast.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { els.toast.hidden = true; }, tone === "error" ? 6000 : 3500);
  }

  // ---------- Adding and removing files ----------

  function isImage(file) {
    return file.type.startsWith("image/") || IMAGE_EXTENSION.test(file.name);
  }

  function fileKey(file) {
    return `${file.name}|${file.size}|${file.lastModified}`;
  }

  function addFiles(fileList) {
    const files = Array.from(fileList || []);
    if (!files.length) return;

    const known = new Set(state.items.map((item) => fileKey(item.file)));
    let notImages = 0;
    let duplicates = 0;
    for (const file of files) {
      if (!isImage(file)) { notImages++; continue; }
      const key = fileKey(file);
      if (known.has(key)) { duplicates++; continue; }
      known.add(key);
      state.items.push({
        id: state.nextId++,
        file,
        url: URL.createObjectURL(file),
        // Files added during a conversion join the running queue.
        status: state.converting ? "queued" : "pending",
        result: null,
        error: null,
        el: null,
      });
    }

    const notes = [];
    if (notImages) notes.push(`${plural(notImages, "file")} skipped: not an image`);
    if (duplicates) notes.push(`${plural(duplicates, "duplicate")} ignored`);
    if (notes.length) showToast(notes.join(" · "));
    renderFiles();
  }

  function removeItem(id) {
    const index = state.items.findIndex((item) => item.id === id);
    if (index < 0) return;
    const [item] = state.items.splice(index, 1);
    URL.revokeObjectURL(item.url);
    renderFiles();
    if (item.status === "done") refreshOutput();

    // Keep keyboard focus in the list instead of losing it to <body>.
    const next = state.items[index] || state.items[index - 1];
    (next ? next.el.querySelector(".remove") : els.dropzone).focus();
  }

  function clearAll() {
    if (state.controller) state.controller.abort();
    state.controller = null;
    state.converting = false;
    for (const item of state.items) URL.revokeObjectURL(item.url);
    state.items = [];
    renderFiles();
    refreshOutput();
    els.dropzone.focus();
  }

  // ---------- Talking to the backend ----------

  async function predict(file, signal) {
    const form = new FormData();
    form.append("image", file, file.name);

    let response;
    try {
      response = await fetch(`${API_BASE}/api/predict`, { method: "POST", body: form, signal });
    } catch (error) {
      if (error.name === "AbortError") throw error;
      throw new NetworkError(`Cannot reach the server at ${API_BASE}. Is backend/app.py running?`);
    }

    let body = null;
    try { body = await response.json(); } catch { /* not JSON */ }
    if (!response.ok) {
      const message = body && body.error && body.error.message;
      throw new Error(message ? capitalize(message) : `The server answered ${response.status}`);
    }
    return body;
  }

  function capitalize(text) {
    return text.charAt(0).toUpperCase() + text.slice(1);
  }

  async function convert() {
    if (state.converting) return;
    const todo = state.items.filter((item) => item.status === "pending" || item.status === "error");
    if (!todo.length) return;

    for (const item of todo) {
      item.status = "queued";
      item.error = null;
    }
    const controller = new AbortController();
    state.controller = controller;
    state.converting = true;
    state.runDone = 0;
    renderFiles();

    // The backend reads one image at a time anyway, so images go one by one and
    // each result shows up as soon as it is ready.
    let item;
    while (!controller.signal.aborted && (item = state.items.find((i) => i.status === "queued"))) {
      item.status = "working";
      renderFiles();
      try {
        item.result = await predict(item.file, controller.signal);
        item.status = "done";
        state.runDone++;
        refreshOutput();
      } catch (error) {
        if (error.name === "AbortError") break;
        item.status = "error";
        item.error = error.message;
        state.runDone++;
        if (error instanceof NetworkError) {
          // Every other request would fail the same way.
          for (const other of state.items) if (other.status === "queued") other.status = "pending";
          showToast(error.message, "error");
          setServer("offline");
          watchServer();
          break;
        }
      }
      renderFiles();
    }

    // Clear may have reset the state while a request was in flight.
    if (state.controller === controller) {
      state.controller = null;
      state.converting = false;
      renderFiles();
    }
  }

  // ---------- Server status ----------

  let serverTimer = 0;

  function setServer(status, text) {
    els.server.dataset.state = status;
    els.server.title = API_BASE;
    els.serverText.textContent = text || {
      checking: "Connecting to the server…",
      online: "Server ready",
      offline: "Server offline",
    }[status];
  }

  async function checkServer() {
    try {
      const response = await fetch(`${API_BASE}/api/model`, { cache: "no-store" });
      if (!response.ok) throw new Error(String(response.status));
      const info = await response.json();
      setServer("online", `Model ${info.run} · ${info.device}`);
      return true;
    } catch {
      setServer("offline", "Server offline — start backend/app.py");
      return false;
    }
  }

  async function watchServer() {
    clearTimeout(serverTimer);
    if (!(await checkServer())) serverTimer = setTimeout(watchServer, SERVER_RETRY_MS);
  }

  // ---------- Rendering: files ----------

  const STATUS = {
    pending: { label: "Ready", tone: "" },
    queued: { label: "Waiting", tone: "" },
    working: { label: "Reading…", tone: "busy" },
    error: { label: "Failed", tone: "err" },
  };

  const BADGE_ICONS = { ok: "check", warn: "warning", err: "error" };

  function warningsFor(result) {
    const warnings = [];
    if (!result.complete) warnings.push("The model did not finish this formula, so it is likely wrong.");
    else if (result.confidence != null && result.confidence < LOW_CONFIDENCE) warnings.push("Low confidence: check the result.");
    if (!result.balanced) warnings.push("The braces do not pair up, so it will not render.");
    return warnings;
  }

  function createRow(item) {
    const row = el("md-list-item", "file");
    const thumb = el("img", "thumb");
    thumb.slot = "start";
    thumb.src = item.url;
    thumb.alt = "";
    thumb.loading = "lazy";

    const name = el("span", "file-name", item.file.name);
    name.slot = "headline";
    name.title = item.file.name;

    const details = el("div", "file-details");
    details.slot = "supporting-text";
    const meta = el("div", "file-meta");
    meta.append(el("span", "size", formatSize(item.file.size)), el("span", "badge"));
    details.append(meta, el("code", "file-latex"), el("p", "file-note"));

    const remove = el("md-icon-button", "remove");
    remove.slot = "end";
    remove.setAttribute("aria-label", `Remove ${item.file.name}`);
    remove.title = "Remove";
    remove.append(icon("close"));
    remove.addEventListener("click", () => removeItem(item.id));

    row.append(thumb, name, details, remove);
    return row;
  }

  function updateRow(item) {
    const row = item.el;
    const badge = row.querySelector(".badge");
    const latex = row.querySelector(".file-latex");
    const note = row.querySelector(".file-note");

    let label;
    let tone;
    let noteText = "";
    let noteTone = "";
    if (item.status === "done") {
      const warnings = warningsFor(item.result);
      const confidence = item.result.confidence;
      label = confidence == null ? "Done" : `${Math.round(confidence * 100)}% confident`;
      tone = warnings.length ? "warn" : "ok";
      noteText = warnings.join(" ");
      noteTone = "warn";
    } else {
      ({ label, tone } = STATUS[item.status]);
      if (item.status === "error") { noteText = item.error; noteTone = "err"; }
    }

    row.dataset.status = item.status;
    row.dataset.tone = tone;
    badge.dataset.tone = tone;
    badge.replaceChildren();
    if (item.status === "working") {
      const spinner = el("md-circular-progress");
      spinner.indeterminate = true;
      spinner.setAttribute("aria-hidden", "true");
      badge.append(spinner);
    } else if (BADGE_ICONS[tone]) {
      badge.append(icon(BADGE_ICONS[tone]));
    }
    badge.append(label);

    const text = item.status === "done" ? item.result.latex : "";
    latex.textContent = text;
    latex.title = text;
    latex.hidden = !text;

    note.textContent = noteText;
    note.dataset.tone = noteTone;
    note.hidden = !noteText;

    row.querySelector(".remove").disabled = item.status === "working";
  }

  function renderFiles() {
    const { items } = state;
    for (const item of items) {
      if (!item.el) item.el = createRow(item);
      updateRow(item);
    }
    // Existing rows are moved, not rebuilt, so thumbnails do not reload.
    els.fileList.replaceChildren(...items.map((item) => item.el));
    els.fileList.hidden = !items.length;
    els.dropzone.classList.toggle("compact", items.length > 0);
    els.fileCount.textContent = items.length ? plural(items.length, "image") : "";

    const todo = items.filter((i) => i.status === "pending" || i.status === "error").length;
    const left = items.filter((i) => i.status === "queued" || i.status === "working").length;
    const done = items.filter((i) => i.status === "done").length;
    if (state.converting) {
      els.convertBtn.textContent = `Converting… ${left} left`;
    } else if (todo && done) {
      els.convertBtn.textContent = `Convert ${todo} more`;
    } else {
      els.convertBtn.textContent = "Convert";
    }
    els.convertBtn.disabled = state.converting || !todo;
    els.clearBtn.disabled = !items.length;
    els.clearBtn.textContent = state.converting ? "Stop & clear" : "Clear";

    // Progress of the current run. Images added during a run join it.
    els.progress.hidden = !state.converting;
    els.progress.value = state.converting ? state.runDone / (state.runDone + left || 1) : 0;
  }

  // ---------- Rendering: result ----------

  /** Rebuild the text from the results. This replaces manual edits in the text box. */
  function refreshOutput() {
    const done = state.items.filter((item) => item.status === "done");
    const key = state.format === "markdown" ? "markdown" : "latex";
    els.output.value = done.map((item) => item.result[key]).join("\n\n");
    renderResult();
  }

  function renderResult() {
    const preview = state.view === "preview";
    for (const tab of els.viewTabs.querySelectorAll("[data-view]")) {
      tab.active = tab.dataset.view === state.view;
    }
    els.formatSelect.value = state.format;
    els.output.hidden = preview;
    els.preview.hidden = !preview;
    if (preview) renderPreview();
    updateResultMeta();
  }

  function updateResultMeta() {
    const text = els.output.value;
    els.copyBtn.disabled = !text.trim();
    els.empty.hidden = state.view !== "text" || text.length > 0;
    const blocks = splitBlocks(text).filter((block) => block.type === "math").length;
    els.resultMeta.textContent = blocks ? plural(blocks, "formula") : "";
  }

  // The preview renders what is in the text box, so edits show up there too.
  function splitBlocks(text) {
    if (state.format === "latex") {
      // One formula per blank-line separated block.
      return text.split(/\n\s*\n/).map((part) => part.trim()).filter(Boolean)
        .map((value) => ({ type: "math", value }));
    }
    // Markdown: $$...$$ and \[...\] are display math, the rest is text with inline $...$.
    const blocks = [];
    const display = /\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]/g;
    let last = 0;
    let match;
    while ((match = display.exec(text))) {
      pushParagraphs(blocks, text.slice(last, match.index));
      blocks.push({ type: "math", value: (match[1] ?? match[2]).trim() });
      last = display.lastIndex;
    }
    pushParagraphs(blocks, text.slice(last));
    return blocks;
  }

  function pushParagraphs(blocks, chunk) {
    for (const part of chunk.split(/\n\s*\n/)) {
      if (part.trim()) blocks.push({ type: "text", value: part.trim() });
    }
  }

  function renderPreview() {
    const root = els.preview;
    root.replaceChildren();
    const text = els.output.value;
    if (!text.trim()) {
      root.append(el("p", "preview-empty", "Nothing to preview yet. Convert some images first."));
      return;
    }
    if (state.katex !== "ready") {
      root.append(el("p", "preview-empty", state.katex === "loading"
        ? "Loading the formula renderer…"
        : "The preview needs KaTeX, which could not be downloaded. Check the internet connection and reload the page. The Text view still works."));
      return;
    }
    for (const block of splitBlocks(text)) {
      root.append(block.type === "math" ? renderDisplay(block.value) : renderParagraph(block.value));
    }
  }

  function renderDisplay(source) {
    const box = el("div", "formula");
    try {
      window.katex.render(source, box, { displayMode: true, throwOnError: true, strict: "ignore" });
    } catch (error) {
      box.className = "formula formula-error";
      box.replaceChildren(el("code", "", source), el("p", "", error.message));
    }
    return box;
  }

  function renderParagraph(text) {
    const paragraph = el("p", "paragraph");
    const inline = /(?<!\\)\$([^$\n]+?)\$/g;
    let last = 0;
    let match;
    while ((match = inline.exec(text))) {
      paragraph.append(text.slice(last, match.index));
      const span = el("span");
      try {
        window.katex.render(match[1], span, { throwOnError: true, strict: "ignore" });
      } catch {
        span.textContent = match[0];
      }
      paragraph.append(span);
      last = inline.lastIndex;
    }
    paragraph.append(text.slice(last));
    return paragraph;
  }

  function loadKatex() {
    const script = document.createElement("script");
    script.src = KATEX_JS;
    script.crossOrigin = "anonymous";
    script.onload = () => {
      state.katex = window.katex ? "ready" : "failed";
      if (window.katex) {
        try {
          window.katex.render(String.raw`\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}`, els.specimen, { displayMode: true });
        } catch { /* keep the plain-text specimen */ }
      }
      renderResult();
    };
    script.onerror = () => { state.katex = "failed"; renderResult(); };
    document.head.append(script);
  }

  // ---------- Copy ----------

  let copyTimer = 0;
  async function copyOutput() {
    const text = els.output.value;
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      if (!legacyCopy(text)) {
        showToast("Copying failed. Select the text and press Ctrl+C.", "error");
        return;
      }
    }
    els.copyLabel.textContent = "Copied";
    els.copyIcon.replaceWith(els.copyIcon = icon("check", "icon"));
    clearTimeout(copyTimer);
    copyTimer = setTimeout(() => {
      els.copyLabel.textContent = "Copy";
      els.copyIcon.replaceWith(els.copyIcon = icon("copy", "icon"));
    }, 1600);
  }

  // For browsers that refuse the async clipboard API (e.g. plain http on another host).
  function legacyCopy(text) {
    const area = el("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.cssText = "position:fixed;top:0;left:0;opacity:0;";
    document.body.append(area);
    area.select();
    let ok = false;
    try { ok = document.execCommand("copy"); } catch { /* unsupported */ }
    area.remove();
    return ok;
  }

  // ---------- Events ----------

  const hasFiles = (event) => Array.from(event.dataTransfer?.types || []).includes("Files");

  function bindUpload() {
    const { dropzone, fileInput } = els;
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        fileInput.click();
      }
    });
    fileInput.addEventListener("change", () => {
      addFiles(fileInput.files);
      fileInput.value = ""; // so picking the same file again still fires "change"
    });

    // dragenter/dragleave also fire for child elements, so count the depth.
    let depth = 0;
    dropzone.addEventListener("dragenter", (event) => {
      if (!hasFiles(event)) return;
      event.preventDefault();
      depth++;
      dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragover", (event) => {
      if (!hasFiles(event)) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
    });
    dropzone.addEventListener("dragleave", () => {
      depth = Math.max(0, depth - 1);
      if (!depth) dropzone.classList.remove("dragover");
    });
    dropzone.addEventListener("drop", (event) => {
      event.preventDefault();
      depth = 0;
      dropzone.classList.remove("dragover");
      addFiles(event.dataTransfer.files);
    });

    // A file dropped next to the zone would otherwise make the browser open it
    // and lose the page.
    for (const type of ["dragover", "drop"]) {
      window.addEventListener(type, (event) => {
        if (hasFiles(event) && !dropzone.contains(event.target)) {
          event.preventDefault();
          event.dataTransfer.dropEffect = "none";
        }
      });
    }

    // Pasted screenshots. Text pasted into the result box keeps working normally.
    document.addEventListener("paste", (event) => {
      const images = Array.from(event.clipboardData?.files || []).filter(isImage);
      if (!images.length) return;
      event.preventDefault();
      const stamp = new Date().toISOString().replace(/\D/g, "").slice(0, 14);
      addFiles(images.map((file, index) => {
        const extension = (file.type.split("/")[1] || "png").replace("jpeg", "jpg");
        const suffix = images.length > 1 ? `-${index + 1}` : "";
        return new File([file], `pasted-${stamp}${suffix}.${extension}`, { type: file.type, lastModified: Date.now() });
      }));
    });
  }

  function bindControls() {
    els.convertBtn.addEventListener("click", convert);
    els.clearBtn.addEventListener("click", clearAll);
    els.copyBtn.addEventListener("click", copyOutput);

    els.viewTabs.addEventListener("change", () => {
      const view = els.viewTabs.activeTab?.dataset.view;
      if (!view || view === state.view) return;
      state.view = view;
      saveSetting("view", state.view);
      renderResult();
    });

    els.formatSelect.addEventListener("change", () => {
      state.format = els.formatSelect.value;
      saveSetting("format", state.format);
      refreshOutput();
    });

    els.output.addEventListener("input", updateResultMeta);

    document.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        convert();
      }
    });
  }

  // ---------- Start ----------

  bindUpload();
  bindControls();
  bindTheme();
  renderFiles();
  refreshOutput();
  setServer("checking");
  watchServer();
  loadKatex();
})();
