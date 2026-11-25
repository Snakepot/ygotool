"use strict";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

// ---------- 零依赖 Markdown 渲染 ----------
function mdToHtml(text) {
  // 先抽代码块
  const blocks = [];
  let s = String(text || "").replace(/```([\s\S]*?)```/g, (_, c) => {
    blocks.push(c.replace(/^\n/, "")); return `\x00CODE${blocks.length - 1}\x00`;
  });
  s = esc(s);
  // 卡牌#id → 可点引用（P2）
  s = s.replace(/卡牌#(\d+)/g, (_, id) =>
    `<a class="cardref" data-id="${id}">卡牌#${id}</a>`);
  // 行内代码
  s = s.replace(/`([^`\n]+)`/g, '<code>$1</code>');
  // 加粗/斜体
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>');
  // 标题
  s = s.replace(/^###\s+(.*)$/gm, '<h3>$1</h3>')
       .replace(/^##\s+(.*)$/gm, '<h2>$1</h2>');
  // 列表
  const lines = s.split("\n");
  let out = [], inUl = false, inOl = false;
  for (const ln of lines) {
    if (/^[-*]\s+/.test(ln)) {
      if (inOl) { out.push("</ol>"); inOl = false; }
      if (!inUl) { out.push("<ul>"); inUl = true; }
      out.push("<li>" + ln.replace(/^[-*]\s+/, "") + "</li>");
    } else if (/^\d+\.\s+/.test(ln)) {
      if (inUl) { out.push("</ul>"); inUl = false; }
      if (!inOl) { out.push("<ol>"); inOl = true; }
      out.push("<li>" + ln.replace(/^\d+\.\s+/, "") + "</li>");
    } else {
      if (inUl) { out.push("</ul>"); inUl = false; }
      if (inOl) { out.push("</ol>"); inOl = false; }
      out.push(ln);
    }
  }
  if (inUl) out.push("</ul>"); if (inOl) out.push("</ol>");
  s = out.join("\n");
  // 段落/换行
  s = s.replace(/\n{2,}/g, "</p><p>").replace(/\n/g, "<br>");
  s = `<p>${s}</p>`;
  // 还原代码块
  s = s.replace(/\x00CODE(\d+)\x00/g, (_, i) => `<pre><code>${esc(blocks[i])}</code></pre>`);
  return s;
}

// ---------- 主题 ----------
const applyTheme = (t) => {
  document.documentElement.dataset.theme = t;
  $("theme-btn").textContent = t === "dark" ? "☀" : "🌙";
  localStorage.theme = t;
};
applyTheme(localStorage.theme || "light");
$("theme-btn").onclick = () => applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");

// ---------- 参数 ----------
const bindParam = (i, v) => { const e = $(i), w = $(v); const u = () => w.textContent = e.value; e.addEventListener("input", u); u(); };
bindParam("p-temp", "v-temp"); bindParam("p-mt", "v-mt"); bindParam("p-fs", "v-fs"); bindParam("p-mc", "v-mc");
fetch("/api/params").then(r => r.json()).then(p => {
  $("p-temp").value = p.temperature; $("v-temp").textContent = p.temperature;
  $("p-mt").value = p.max_tokens; $("v-mt").textContent = p.max_tokens;
  $("p-mc").value = p.max_context; $("v-mc").textContent = p.max_context;
  $("v-model").textContent = p.model;
  $("backend-tag").textContent = p.backend + (p.backend === "mock" ? "(未配.env)" : "");
  $("v-index").textContent = p.has_index ? "已建" : "未建";
});
$("collapse-btn").onclick = () => $("params").classList.toggle("collapsed");

// ---------- 卡牌浏览（模态） ----------
async function showCard(id) {
  try {
    const r = await fetch(`/api/card/${id}?faq=1`);
    if (!r.ok) { alert("未找到该卡"); return; }
    const c = await r.json();
    const first = (c.cn_name || "?").trim().charAt(0) || "?";
    let h = `<div class="modal-head"><span class="thumb modal-thumb">${esc(first)}</span>`
      + `<div><h3>${esc(c.cn_name)}</h3><div class="types">#${esc(String(c.id))} · ${esc(c.types)}</div></div></div>`;
    if (c.pdesc) h += `<h3>灵摆效果</h3><pre>${esc(c.pdesc)}</pre>`;
    if (c.desc) h += `<h3>效果</h3><pre>${esc(c.desc)}</pre>`;
    if (c.faqcount) {
      h += `<h3>官方 FAQ（${c.faqcount} 条）</h3>`;
      c.faqs.forEach(f => h += `<div class="faq-item"><div class="q">Q: ${esc(f.q)}</div><div class="a">A: ${esc(f.a)}</div></div>`);
    }
    $("modal-body").innerHTML = h;
    $("card-modal").hidden = false;
  } catch (e) { alert("加载失败: " + e); }
}
$("modal-close").onclick = () => $("card-modal").hidden = true;
document.querySelector(".modal-backdrop").onclick = () => $("card-modal").hidden = true;

function renderCardItem(container, c) {
  const d = document.createElement("div");
  d.className = "card-item";
  const first = (c.cn_name || "?").trim().charAt(0) || "?";
  d.innerHTML = `<span class="thumb">${esc(first)}</span>`
    + `<span class="card-main"><span class="name">${esc(c.cn_name)}</span>`
    + `<span class="sub">#${esc(String(c.id))} ${esc(c.types || "")}</span>`
    + (c.snippet ? `<span class="snippet">${esc(c.snippet)}</span>` : "")
    + `</span>`;
  d.onclick = () => showCard(c.id);
  container.appendChild(d);
}
// 加载骨架 / 空态（纯前端占位，不依赖后端）
function showSkeleton(el, rows = 6) {
  el.innerHTML = "";
  for (let i = 0; i < rows; i++) {
    const d = document.createElement("div");
    d.className = "skeleton";
    el.appendChild(d);
  }
}
function emptyState(el, text) {
  el.innerHTML = `<div class="empty"><span class="empty-icon">🔍</span><span>${esc(text)}</span></div>`;
}
async function doSearch() {
  const q = $("search-input").value.trim(); if (!q) return;
  const el = $("card-list"); showSkeleton(el, 5);
  const d = await (await fetch(`/api/search?q=${encodeURIComponent(q)}&limit=30`)).json();
  el.innerHTML = "";
  d.result.forEach(c => renderCardItem(el, c));
  if (!d.result.length) emptyState(el, "没有匹配的卡牌，试试别的关键词");
}
async function doEffectSearch() {
  const q = $("effect-input").value.trim(); if (!q) return;
  const el = $("effect-list"); showSkeleton(el, 5);
  const reason = $("effect-reason").value;
  const url = `/api/action-search?q=${encodeURIComponent(q)}` + (reason ? `&reason=${reason}` : "");
  const d = await (await fetch(url)).json();
  el.innerHTML = "";
  const mode = d.mode === "effect" ? "子串回退" : (reason ? `仅${reason}段` : "本体");
  const tier = (title, cards) => {
    if (!cards.length) return;
    const h = document.createElement("div"); h.className = "sub";
    h.innerHTML = `<b>${esc(title)}（${cards.length}）</b>`;
    el.appendChild(h); cards.forEach(c => renderCardItem(el, c));
  };
  const tOf = (c) => c.tier !== undefined ? c.tier : (c.specific ? 2 : 0);
  const all = d.modern.concat(d.uncertain);
  tier(`★ 精确匹配（${mode}）`, all.filter(c => tOf(c) === 2));
  tier("◆ 次精确（限定一致，来源更宽）", all.filter(c => tOf(c) === 1));
  tier("通用 · 现代", d.modern.filter(c => tOf(c) === 0));
  tier("通用 · 未确认（90X 前旧措辞，须查FAQ）", d.uncertain.filter(c => tOf(c) === 0));
  if (!d.count) emptyState(el, "没有匹配的卡牌，试试换一种说法或补上限定词");
}
$("search-btn").onclick = doSearch;
$("effect-btn").onclick = doEffectSearch;
$("effect-clear").onclick = () => { $("effect-input").value = ""; $("effect-list").innerHTML = ""; };
$("search-clear").onclick = () => { $("search-input").value = ""; $("card-list").innerHTML = ""; };
$("search-input").addEventListener("keydown", e => { if (e.key === "Enter") doSearch(); });
$("effect-input").addEventListener("keydown", e => { if (e.key === "Enter") doEffectSearch(); });

// 按系列搜（卡名全量）
async function doSeriesSearch() {
  const q = $("series-input").value.trim(); if (!q) return;
  const el = $("series-list"); showSkeleton(el, 4);
  const d = await (await fetch(`/api/series-search?q=${encodeURIComponent(q)}`)).json();
  el.innerHTML = "";
  if (!d.count) { emptyState(el, `未识别系列（token: ${d.token || "无"}）`); return; }
  const h = document.createElement("div"); h.className = "sub";
  h.innerHTML = `<b>系列「${esc(d.token)}」（${d.count}）</b>`;
  el.appendChild(h);
  d.cards.forEach(c => renderCardItem(el, c));
}
$("series-btn").onclick = doSeriesSearch;
$("series-clear").onclick = () => { $("series-input").value = ""; $("series-list").innerHTML = ""; };
$("series-input").addEventListener("keydown", e => { if (e.key === "Enter") doSeriesSearch(); });

// 答案里卡牌#id 可点（事件委托）
$("messages").addEventListener("click", e => {
  const a = e.target.closest("a.cardref");
  if (a) showCard(a.dataset.id);
});

// ---------- 多轮聊天 ----------
let convo = [];          // [{role, content}] 历次完整轮次
let abortCtrl = null;

function curParams() {
  return { task: $("task").value, persona: $("persona").value,
    temperature: parseFloat($("p-temp").value), max_tokens: parseInt($("p-mt").value),
    num_few_shots: parseInt($("p-fs").value), max_context: parseInt($("p-mc").value),
    thinking: $("p-thinking").checked };
}

async function sendChat() {
  const q = $("question").value.trim(); if (!q) return;
  const p = curParams();
  $("send-btn").disabled = true; $("stop-btn").hidden = false;
  abortCtrl = new AbortController();

  // 渲染用户消息
  const msgs = $("messages");
  const u = document.createElement("div"); u.className = "msg user";
  u.innerHTML = `<div class="body">${mdToHtml(q)}</div>`;
  msgs.appendChild(u);
  $("question").value = "";

  // bot 占位
  const b = document.createElement("div"); b.className = "msg bot";
  b.innerHTML = `<div class="role">assistant · ${p.task} · ${p.persona === "teach" ? "教学" : "裁定官"}</div>`
    + `<div class="trace"></div><details class="reasoning"><summary>思考过程</summary><div class="rbody"></div></details>`
    + `<div class="body"></div><div class="usage"></div>`;
  msgs.appendChild(b);
  const traceEl = b.querySelector(".trace");
  const rDetails = b.querySelector("details.reasoning");
  const rBody = b.querySelector(".rbody");
  const bodyEl = b.querySelector(".body");
  const usageEl = b.querySelector(".usage");
  let contentAcc = "", reasonAcc = "";

  try {
    const resp = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, ...p, history: convo.slice(-6) }),
      signal: abortCtrl.signal,
    });
    const reader = resp.body.getReader(); const dec = new TextDecoder(); let buf = "";
    while (true) {
      const { done, value } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, idx); buf = buf.slice(idx + 2);
        const line = block.split("\n").find(l => l.startsWith("data:"));
        if (!line) continue;
        const ev = JSON.parse(line.slice(5).trim());
        if (ev.type === "trace") {
          const chips = (ev.tools || []).map(t => `<span class="chip">${t}</span>`).join("");
          if (ev.budget) {
            // P1-1：mock 与 ReAct 路径的 trace 现在同形（都带 budget/iterations/llm_usage），统一展示
            const bk = ev.budget;
            const it = ev.iterations ? ` · 迭代${ev.iterations}轮` : "";
            traceEl.innerHTML = `${chips} 素材${ev.retrieved_total||0}条${it} | 预算 ${bk.tokens_total||0}/${bk.max_context} ${bk.within_budget ? "✓" : "超"}`;
          } else {
            // 兜底：仅防御旧服务端；正常前端收到的 trace 一定带 budget
            const it = ev.iterations ? ` · 迭代${ev.iterations}轮` : "";
            traceEl.innerHTML = `${chips}${it} ${ev.retrieved_total!=null ? "素材"+ev.retrieved_total+"条" : ""}`;
          }
        } else if (ev.type === "reasoning") {
          reasonAcc += ev.text; rBody.textContent = reasonAcc;
          if (!rDetails.open) rDetails.querySelector("summary").classList.add("thinking-dot");
        } else if (ev.type === "content") {
          contentAcc += ev.text;
          bodyEl.innerHTML = mdToHtml(contentAcc);
          rDetails.querySelector("summary")?.classList.remove("thinking-dot");
        } else if (ev.type === "usage") {
          const u2 = ev.usage || {};
          usageEl.textContent = `prompt ${u2.prompt} + completion ${u2.completion} = ${u2.total}`;
        } else if (ev.type === "error") {
          bodyEl.innerHTML = `<span style="color:#c00">错误: ${esc(ev.message)}</span>`;
        }
        bodyEl.scrollIntoView({ behavior: "smooth", block: "end" });
      }
    }
    if (contentAcc) convo.push({ role: "user", content: q }, { role: "assistant", content: contentAcc });
  } catch (e) {
    if (e.name === "AbortError") bodyEl.innerHTML = "<span style='color:#888'>已停止</span>";
    else bodyEl.innerHTML = `<span style="color:#c00">流式读取错误: ${esc(String(e))}</span>`;
  }
  $("send-btn").disabled = false; $("stop-btn").hidden = true; abortCtrl = null;
}

$("send-btn").onclick = sendChat;
$("stop-btn").onclick = () => abortCtrl && abortCtrl.abort();
$("clear-btn").onclick = () => { convo = []; $("messages").innerHTML = ""; };
$("question").addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); }
});
