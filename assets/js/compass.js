/* Browser router: embed the request with the model Python used, vote over the nearest benchmark
   prompts, show the task's best model with proof. Mirrors the Python router (see /method/). */
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6";
const EMBED_MODEL = "Xenova/all-MiniLM-L6-v2";
const LOAD_TIMEOUT_MS = 45000;
// Confidence gate defaults; the real floors come from router.json eval (exported by Python).
const DEFAULT_MIN_SIM = 0.35;
const DEFAULT_MIN_SHARE = 0.4;
// Out of scope (Compass only covers images generated from text); checked before the model loads.
const OUT_OF_SCOPE = [
  "video", "videos", "animate", "animated", "animation", "gif",
  "remove the background", "remove background", "background removal", "cut out the background",
  "upscale", "upscaling", "enhance", "sharpen", "restore", "retouch",
  "edit my", "edit this", "edit the", "edit a photo", "fix my photo", "inpaint", "outpaint",
  "poem", "essay", "lyrics", "song", "write me", "write a", "text only", "caption for",
  "music", "audio", "voice", "voiceover", "translate",
];

const base = document.documentElement.dataset.base || "";
const url = (p) => `${base}/${String(p).replace(/^\/+/, "")}`;
const $ = (id) => document.getElementById(id);
const form = $("ask");
const input = $("prompt");
const status = $("status");
const utm = JSON.parse(document.body.dataset.utm || "{}");

let dataPromise = null;
let extractorPromise = null;

function loadData() {
  if (!dataPromise) {
    dataPromise = Promise.all([
      fetch(url("data/compass.json")).then((r) => r.json()),
      fetch(url("data/router.json")).then((r) => r.json()),
    ]).then(([compass, router]) => ({ compass, router }));
  }
  return dataPromise;
}

function loadExtractor() {
  if (!extractorPromise) {
    extractorPromise = (async () => {
      const { pipeline, env } = await import(TRANSFORMERS_URL);
      env.allowLocalModels = false;
      return pipeline("feature-extraction", EMBED_MODEL, { dtype: "q8" });
    })();
    extractorPromise.catch(() => { extractorPromise = null; });
  }
  return extractorPromise;
}

function withTimeout(promise, ms, label) {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(`${label} timed out`)), ms);
    promise.then((v) => { clearTimeout(t); resolve(v); }, (e) => { clearTimeout(t); reject(e); });
  });
}

async function embed(extractor, text) {
  // ONE text per call: batches drift with the int8 model (parity doc §4).
  const out = await extractor(text, { pooling: "mean", normalize: true });
  return Array.from(out.data);
}

export function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

export function route(queryVec, prompts, k) {
  const sims = prompts.map((p, i) => ({ i, sim: cosine(queryVec, p.embedding) }));
  sims.sort((a, b) => b.sim - a.sim || a.i - b.i);
  const top = sims.slice(0, Math.max(1, Math.min(k, sims.length)));
  const totals = new Map(), firstSeen = new Map();
  top.forEach(({ i, sim }, pos) => {
    const task = prompts[i].task;
    totals.set(task, (totals.get(task) ?? 0) + Math.max(sim, 0));
    if (!firstSeen.has(task)) firstSeen.set(task, pos);
  });
  const sum = [...totals.values()].reduce((a, b) => a + b, 0);
  const scores = new Map([...totals].map(([t, v]) => [t, sum > 0 ? v / sum : 1 / totals.size]));
  let best = null;
  for (const [t, s] of scores) {
    if (best === null || s > scores.get(best) || (s === scores.get(best) && firstSeen.get(t) < firstSeen.get(best))) best = t;
  }
  return { task: best, scores: Object.fromEntries(scores), neighbors: top.map(({ i, sim }) => ({ ...prompts[i], sim })) };
}

export function keywordRoute(query, fallback) {
  const words = query.toLowerCase().match(/[a-z0-9][a-z0-9'+-]*/g) ?? [];
  const q = " " + words.join(" ") + " ";
  const qWords = new Set(words);
  let best = null, bestScore = 0;
  for (const [task, keywords] of Object.entries(fallback)) {
    let score = 0;
    for (const kw of keywords) {
      if (kw.includes(" ")) { if (q.includes(" " + kw + " ")) score += 2; }
      else if (qWords.has(kw)) score += 1;
    }
    if (score > bestScore) { best = task; bestScore = score; }
  }
  return best;
}

export function outOfScope(query) {
  const words = query.toLowerCase().match(/[a-z0-9][a-z0-9'+-]*/g) ?? [];
  const q = " " + words.join(" ") + " ";
  const qWords = new Set(words);
  return OUT_OF_SCOPE.some((kw) => (kw.includes(" ") ? q.includes(" " + kw + " ") : qWords.has(kw)));
}

/* Mirrors router.is_confident: nearest prompt close enough AND a clear vote. */
export function isConfident(topSim, taskShare, minSim, minShare) {
  return topSim >= minSim && taskShare >= minShare;
}

/* Close race: the runner-up's 95 % interval overlaps the best's (rank.is_close_race). */
export function closeRace(ranking) {
  if (!ranking || ranking.length < 2) return false;
  const [a, b] = ranking;
  return a.ci_high >= b.ci_low && a.ci_low <= b.ci_high;
}

function outbound(model, taskId) {
  const q = new URLSearchParams({
    utm_source: utm.source || "model-compass",
    utm_medium: utm.medium || "referral",
    utm_campaign: utm.campaign || "open-doors",
    utm_content: `${taskId}-${model.id}`,
  });
  return model.deep_link + (model.deep_link.includes("?") ? "&" : "?") + q.toString();
}

function setStatus(text) { status.textContent = text || ""; }

/* First sentence of a "why" (model names like "GPT Image 2.5" do not split it). */
export function firstSentence(text) {
  const t = String(text || "").trim();
  const m = t.match(/^[\s\S]*?[.!?](?=\s+[A-Z“"(]|$)/);
  return m ? m[0].trim() : t;
}

/* 'Best AI model for logos' -> 'logos' (pages/task.py headline_noun). */
export function headlineNoun(task) {
  const head = String(task.headline || "");
  const prefix = "Best AI model for ";
  return head.startsWith(prefix) ? head.slice(prefix.length).trim() : String(task.name || "").toLowerCase();
}

/* Verdict stat row + short "why" line (mirrors pages/task.py). */
export function verdictStats(task, top, runner) {
  const stats = [];
  if (!top) return stats;
  stats.push({ value: `${Math.round(top.win_prob_vs_field * 100)}%`, label: "chance of ranking first" });
  if (runner) stats.push({ value: runner.name, label: "runner-up" });
  if (task.needs_text && top.ocr_accuracy != null) stats.push({ value: `${Math.round(top.ocr_accuracy * 100)}%`, label: "text spelled right" });
  return stats;
}

export function shortWhy(task, top, wins) {
  if (!top) return "";
  const n = task.n_prompts || 0;
  if (task.needs_text && top.ocr_accuracy != null && n) return `Judge's most likely first pick; text spelled right on ${Math.round(top.ocr_accuracy * n)} of ${n}.`;
  if (wins && n) return `Judge's most likely first pick; first on ${wins} of ${n} requests.`;
  return "Judge's most likely first pick on this task's requests.";
}

/* Requests where the model had the best mean judge rank; ties share the win. */
export function promptWins(prompts, modelId) {
  let wins = 0;
  for (const p of prompts) {
    const ranked = p.outputs.filter((o) => o.judge_rank_mean != null);
    if (!ranked.length) continue;
    const best = Math.min(...ranked.map((o) => o.judge_rank_mean));
    if (ranked.some((o) => o.model === modelId && o.judge_rank_mean === best)) wins += 1;
  }
  return wins;
}

function render({ compass, router }, taskId, neighbors, mode, scores) {
  const task = compass.tasks.find((t) => t.id === taskId);
  const rec = (task && task.recommendation) || router.task_recommendation[taskId];
  if (!task || !rec || !rec.best) { showPickTask(); return; }
  const models = Object.fromEntries(compass.models.map((m) => [m.id, m]));
  const best = models[rec.best] || { id: rec.best, name: rec.best, provider: "", tier: "standard", deep_link: "#" };
  const taskUrl = url(`best-ai-model-for/${task.slug}/`);

  $("verdict-task").textContent = `Best for ${headlineNoun(task)}`;
  $("verdict-title").textContent = best.name;
  $("verdict-provider").textContent = best.provider;
  const tier = $("verdict-tier");
  tier.textContent = best.tier;
  tier.className = `tier tier--${best.tier}`;
  const ranking = task.ranking || [];
  const top = ranking.find((r) => r.model === best.id) || ranking[0];
  const runnerModel = rec.runner_up && models[rec.runner_up];
  const taskPrompts = compass.prompts.filter((p) => p.task === task.id);
  const statsEl = $("verdict-stats");
  if (statsEl) {
    statsEl.replaceChildren(...verdictStats(task, top, runnerModel).map((st) => {
      const li = document.createElement("li");
      li.append(Object.assign(document.createElement("strong"), { textContent: st.value }), Object.assign(document.createElement("span"), { textContent: st.label }));
      return li;
    }));
  }
  $("verdict-why").textContent = top ? shortWhy(task, top, promptWins(taskPrompts, best.id)) : firstSentence(rec.why || task.description);
  const close = closeRace(task.ranking);
  const badge = $("verdict-close");
  if (badge) badge.hidden = !close;
  const go = $("verdict-go");
  go.textContent = `Make it in Picsart with ${best.name}`;
  go.href = outbound(best, task.id);
  go.dataset.out = `${task.id}/${best.id}`;
  const alt = $("verdict-alt");
  const runner = rec.runner_up && models[rec.runner_up];
  if (alt) {
    if (close && runner && runner.id !== best.id) {
      alt.textContent = `Or make it with ${runner.name}`;
      alt.href = outbound(runner, task.id);
      alt.dataset.out = `${task.id}/${runner.id}`;
      alt.hidden = false;
    } else {
      alt.hidden = true;
    }
  }
  const tl = $("verdict-task-link");
  tl.textContent = `See all ${task.name.toLowerCase()} results`;
  tl.href = taskUrl;

  const fb = $("fallback");
  if (rec.fallback_cheaper && models[rec.fallback_cheaper] && rec.fallback_cheaper !== best.id) {
    const cheaper = models[rec.fallback_cheaper];
    $("fallback-text").textContent = `${cheaper.name} (${cheaper.provider}) ties ${best.name} within the error bars on this task and uses fewer credits per image.`;
    const fgo = $("fallback-go");
    fgo.textContent = `Open ${cheaper.name} instead`;
    fgo.href = outbound(cheaper, task.id);
    fgo.dataset.out = `${task.id}/${cheaper.id}`;
    fb.hidden = false; fb.open = false;
  } else {
    fb.hidden = true;
  }

  const bits = [];
  if (mode === "keyword") bits.push("Task found by keywords; the router model did not load");
  else if (scores && scores[task.id] != null) bits.push(`Task match ${Math.round(scores[task.id] * 100)}% of the ${neighbors.length} nearest tested requests`);
  if (task.judge_agreement_tau != null) bits.push(`Judge agreement ${Number(task.judge_agreement_tau).toFixed(2)}`);
  $("verdict-confidence").textContent = bits.join(". ") + (bits.length ? "." : "");

  const proof = $("proof");
  proof.innerHTML = "";
  const byId = Object.fromEntries(compass.prompts.map((p) => [p.id, p]));
  const seen = new Set();
  const items = [];
  const pool = [...neighbors.filter((n) => n.task === task.id), ...(compass.prompts.filter((p) => p.task === task.id))];
  for (const n of pool) {
    if (items.length >= 4 || seen.has(n.id)) continue;
    const p = byId[n.id];
    const o = p && p.outputs.find((x) => x.model === best.id);
    if (!o) continue;
    seen.add(n.id);
    items.push({ p, o });
  }
  items.slice(0, Math.max(2, Math.min(4, items.length))).forEach(({ p, o }, index) => {
    const li = document.createElement("li");
    li.className = "proof__item";
    const a = document.createElement("a");
    a.href = url(`compare/?prompt=${encodeURIComponent(p.id)}`);
    const img = document.createElement("img");
    img.className = "proof__img";
    img.loading = index < 2 ? "eager" : "lazy";
    img.decoding = "async";
    img.width = 512; img.height = 512;
    img.src = url(o.thumb);
    img.alt = `${best.name} output for: ${p.text}`;
    a.appendChild(img);
    const cap = document.createElement("p");
    cap.className = "proof__cap";
    cap.textContent = p.text;
    li.append(a, cap);
    proof.appendChild(li);
  });
  $("verdict-proof-title") && ($("verdict-proof-title").hidden = items.length === 0);

  $("pick-task").hidden = true;
  const v = $("verdict");
  v.hidden = false;
  v.scrollIntoView({ behavior: "smooth", block: "nearest" });
  if (window.compassCount) window.compassCount(`route/${task.id}/${best.id}/${mode}`);
}

const PICK_MESSAGES = {
  unknown: "Compass could not tell what you want to make. Pick the closest task:",
  unsure: "Compass is not sure. Pick the closest text-to-image task:",
  scope: "Compass covers images made from text, not video, photo editing, upscaling or writing. Want an image? Pick a task:",
  error: "Compass could not load its data. Reload the page or pick a task below:",
};

/* reason: unknown | unsure | scope | error; nearest: [{name, share}] when unsure. */
function showPickTask(reason = "unknown", nearest = []) {
  $("verdict").hidden = true;
  const box = $("pick-task");
  const text = $("pick-task-text");
  if (text) text.textContent = PICK_MESSAGES[reason] || PICK_MESSAGES.unknown;
  const near = $("pick-task-near");
  if (near) {
    if (reason === "unsure" && nearest.length) {
      near.textContent = "Closest: " + nearest.map((n) => `${n.name} (${Math.round(n.share * 100)}% of the vote)`).join(", ") + ".";
      near.hidden = false;
    } else {
      near.hidden = true;
    }
  }
  box.dataset.reason = reason;
  box.hidden = false;
  box.scrollIntoView({ behavior: "smooth", block: "nearest" });
  if (window.compassCount) window.compassCount(`route/none/${reason}`);
}

function nearestTasks(compass, scores, n = 2) {
  return Object.entries(scores || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, n)
    .map(([id, share]) => ({ name: (compass.tasks.find((t) => t.id === id) || { name: id }).name, share }));
}

async function ask(text) {
  const q = text.trim();
  if (!q) return;
  if (outOfScope(q)) { setStatus(""); showPickTask("scope"); form.classList.add("is-settled"); return; }
  form.classList.add("is-loading");
  form.classList.remove("is-settled");
  try {
    const data = await loadData();
    const gate = data.router.eval || {};
    const minSim = gate.min_similarity ?? DEFAULT_MIN_SIM;
    const minShare = gate.min_task_share ?? DEFAULT_MIN_SHARE;
    let mode = "embed";
    let taskId = null, neighbors = [], scores = null;
    try {
      setStatus("Loading the router (23 MB, once, cached by your browser)…");
      const extractor = await withTimeout(loadExtractor(), LOAD_TIMEOUT_MS, "router load");
      setStatus("Reading your request…");
      const vec = await embed(extractor, q);
      const r = route(vec, data.router.prompts, data.router.k);
      taskId = r.task; neighbors = r.neighbors; scores = r.scores;
      const topSim = neighbors.length ? neighbors[0].sim : 0;
      window.__lastRoute = { task: taskId, topSim, share: scores[taskId], scores };
      if (!isConfident(topSim, scores[taskId] ?? 0, minSim, minShare)) {
        setStatus("");
        showPickTask("unsure", nearestTasks(data.compass, scores));
        return;
      }
    } catch (err) {
      console.warn("router model unavailable, using keywords:", err);
      mode = "keyword";
      taskId = keywordRoute(q, data.router.keyword_fallback);
    }
    setStatus("");
    if (!taskId) { showPickTask("unknown"); return; }
    render(data, taskId, neighbors, mode, scores);
  } catch (err) {
    console.error(err);
    setStatus("");
    showPickTask("error");
  } finally {
    form.classList.remove("is-loading");
    form.classList.add("is-settled");
  }
}

if (form) {
  form.addEventListener("submit", (ev) => { ev.preventDefault(); ask(input.value); });
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); ask(input.value); }
  });
  input.addEventListener("focus", () => { loadData(); }, { once: true });
  document.querySelectorAll(".chip[data-example]").forEach((chip) => {
    chip.addEventListener("click", () => { input.value = chip.dataset.example; ask(input.value); });
  });
  const params = new URLSearchParams(location.search);
  const q = params.get("q");
  if (q) { input.value = q; ask(q); }
  if (params.get("selftest") === "1") selfTest();
}

/* ?selftest=1: embed the parity fixture texts and compare with Python's vectors. */
async function selfTest() {
  const panel = document.createElement("pre");
  panel.id = "selftest";
  panel.style.cssText = "margin:16px 0;padding:12px;border:1px solid var(--line);border-radius:6px;font-size:12px;overflow:auto;background:var(--surface)";
  form.after(panel);
  const log = (s) => { panel.textContent += s + "\n"; console.log("[selftest]", s); };
  try {
    const fixture = await (await fetch(url("test/router_parity.json"))).json();
    log(`parity fixture: ${fixture.items.length} texts, min cosine ${fixture.min_cosine}`);
    const extractor = await loadExtractor();
    const results = [];
    for (const item of fixture.items) {
      const mine = await embed(extractor, item.text);
      const cos = cosine(mine, item.embedding);
      results.push({ text: item.text, cosine: cos, ok: cos > fixture.min_cosine });
      log(`${cos > fixture.min_cosine ? "PASS" : "FAIL"} ${cos.toFixed(5)}  ${item.text}`);
    }
    const min = Math.min(...results.map((r) => r.cosine));
    log(`min cosine ${min.toFixed(5)} ${min > fixture.min_cosine ? "(all pass)" : "(FAIL)"}`);
    window.__parity = { min, results, done: true };
  } catch (err) {
    log(`selftest error: ${err}`);
    window.__parity = { error: String(err), done: true };
  }
}
