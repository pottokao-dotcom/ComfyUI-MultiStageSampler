// Multi-Stage Sampler in-node recipe editor: stage table, flow bar + cost estimate, save / set default, merge, live re-run. UI follows Comfy.Locale.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CUSTOM = "custom";

// 介面文字跟著 ComfyUI 的語言設定(設定 → Comfy.Locale),沒有的語言用英文
const I18N = {
  en: { recipe: "Recipe", est: (t, r) => ` est. ${t}s (QI2.1 NVFP4; 25-step ≈ ${r}s)`, model: "Model", sched: "Schedule / σ", noise: "+noise",
        sigph: "σ, comma separated", addBase: "+ base (copy schedule)", addFast: "+ fast", addRefine: "+ refine (renoise finish)",
        save: "💾 Save as", setDef: "⭐ Set default", askName: "Recipe name", saved: (n) => `Saved "${n}" (reload the node list to see it in preset)`,
        saveFail: "Save failed: ", savedDef: (n) => `"${n}" saved and set as default`, fail: "Failed: ", live: " Re-run on every change (set seed to fixed)",
        empty: "No stages: add one with + above", merge: "Merge ", mAll: "append whole recipe", mLast: "append its last stage", mReplace: "replace my last stage",
        mBtn: "🔗 Merge", mWarn: "⚠ The merged part starts again from step 0 of a schedule (= redraws from scratch); usually continue from later steps or use σ / refine",
        edited: " (edited)", mine: "My recipe" },
  "zh-TW": { recipe: "配方", est: (t, r) => ` 估 ${t}s(QI2.1 NVFP4;原 25 步 ≈ ${r}s)`, model: "模型", sched: "排程/σ", noise: "加雜訊",
        sigph: "σ,逗號分隔", addBase: "+ base(照抄排程)", addFast: "+ fast", addRefine: "+ refine(加雜訊收尾)",
        save: "💾 另存配方", setDef: "⭐ 設為預設", askName: "配方名稱", saved: (n) => `已存「${n}」(重新整理節點清單後會出現在配方下拉)`,
        saveFail: "存檔失敗:", savedDef: (n) => `「${n}」已存並設為預設`, fail: "失敗:", live: " 改了就出圖(seed 請設 fixed)",
        empty: "沒有任何段:按上面的 + 新增", merge: "合併 ", mAll: "整份接在後面", mLast: "只接它的最後一段", mReplace: "取代我的最後一段",
        mBtn: "🔗 合併", mWarn: "⚠ 接進來的第一段是從第 0 步照抄排程 = 重新從頭畫,通常要改成接續的步數或用 σ/refine",
        edited: "(改)", mine: "我的配方" },
  zh: { recipe: "配方", est: (t, r) => ` 估 ${t}s(QI2.1 NVFP4;原 25 步 ≈ ${r}s)`, model: "模型", sched: "排程/σ", noise: "加噪声",
        sigph: "σ,逗号分隔", addBase: "+ base(照抄排程)", addFast: "+ fast", addRefine: "+ refine(加噪声收尾)",
        save: "💾 另存配方", setDef: "⭐ 设为默认", askName: "配方名称", saved: (n) => `已存「${n}」(刷新节点列表后会出现在配方下拉)`,
        saveFail: "保存失败:", savedDef: (n) => `「${n}」已存并设为默认`, fail: "失败:", live: " 改了就出图(seed 请设 fixed)",
        empty: "没有任何段:按上面的 + 新增", merge: "合并 ", mAll: "整份接在后面", mLast: "只接它的最后一段", mReplace: "取代我的最后一段",
        mBtn: "🔗 合并", mWarn: "⚠ 接进来的第一段是从第 0 步照抄排程 = 重新从头画,通常要改成接续的步数或用 σ/refine",
        edited: "(改)", mine: "我的配方" },
};
function lang() {
  let l = null;
  try { l = app.ui?.settings?.getSettingValue?.("Comfy.Locale"); } catch (e) { }
  try { l = l || app.extensionManager?.setting?.get?.("Comfy.Locale"); } catch (e) { }
  l = l || navigator.language || "en";
  if (I18N[l]) return I18N[l];
  if (/^zh[-_](TW|HK|Hant)/i.test(l)) return I18N["zh-TW"];
  if (/^zh/i.test(l)) return I18N.zh;
  return I18N.en;
}
const COST = { fixed: 1.4, base: 0.59, fast: 0.81, stage: 0.2 };   // QI2.1 NVFP4 @ 5060 Ti 實測回歸(bench qi21_v1)
const SAMPLERS = ["euler", "euler_ancestral", "heun", "dpmpp_2m", "dpmpp_sde", "res_multistep", "lcm"];
const COLORS = { base: "#8a8f98", fast: "#e0913a", refine: "#4a90d9" };

const css = `
.ms-ed{font:12px sans-serif;color:#ddd;background:#1c1c1c;border:1px solid #333;border-radius:6px;padding:6px;box-sizing:border-box;width:100%;overflow:auto}
.ms-ed table{border-collapse:collapse;width:100%}.ms-ed td,.ms-ed th{padding:2px 3px;text-align:left;white-space:nowrap}
.ms-ed th{color:#999;font-weight:normal}.ms-ed input,.ms-ed select{background:#111;color:#eee;border:1px solid #444;border-radius:3px;font:12px sans-serif;padding:1px 3px}
.ms-ed input[type=number]{width:44px}.ms-ed input.sig{width:120px}.ms-ed select{max-width:92px}.ms-ed button{background:#2a2a2a;color:#ddd;border:1px solid #555;border-radius:4px;padding:2px 7px;cursor:pointer;margin:2px}.ms-ed td button{padding:0 4px;margin:0 1px}
.ms-ed button:hover{border-color:#d9a441}.ms-bar{display:flex;height:16px;margin:6px 0 2px;border-radius:3px;overflow:hidden}
.ms-bar div{height:100%;border-right:1px solid #1c1c1c;font-size:10px;color:#111;text-align:center;overflow:hidden}
.ms-row{display:flex;flex-wrap:wrap;align-items:center;gap:4px;margin-top:4px}.ms-err{color:#e06c5a}.ms-note{color:#999}`;
if (!document.getElementById("ms-ed-css")) { const st = document.createElement("style"); st.id = "ms-ed-css"; st.textContent = css; document.head.appendChild(st); }

function kind(s) { return s.noise ? "refine" : s.model; }
function nsteps(s) { return s.scheduler ? (s.range[1] - s.range[0]) : s.sigmas.length - (s.noise ? 1 : 0); }
function evals(s) { return nsteps(s) * ((s.cfg || 1) > 1 ? 2 : 1); }
function estimate(stages) {
  let t = COST.fixed + COST.stage * stages.length;
  for (const s of stages) t += evals(s) * (s.model === "fast" ? COST.fast : COST.base);
  return t;
}

app.registerExtension({
  name: "pottokao.MultiStageSampler.editor",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "MultiStageSampler") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      const node = this;
      const W = (n) => node.widgets.find((w) => w.name === n);
      const el = document.createElement("div"); el.className = "ms-ed";
      const T = lang();
      let lib = { recipes: [], default: null }, cur = { name: T.mine, stages: [] }, live = false, timer = null;

      const load = async () => { try { lib = await (await api.fetchApi("/multistage/recipes")).json(); } catch (e) { lib = { recipes: [] }; } };
      const fromWidgets = () => {
        const p = W("preset")?.value;
        const rc = p && !p.startsWith("custom") ? lib.recipes.find((x) => x.name === p) : null;
        if (rc) cur = JSON.parse(JSON.stringify(rc));
        else { try { const j = JSON.parse(W("recipe").value || "{}"); if (j.stages) cur = j; } catch (e) { } }
      };
      const push = () => {                                   // 編輯器 → recipe 框,preset 切到 custom
        if (lib.recipes.some((x) => x.name === cur.name)) { cur.name = cur.name + T.edited; setTimeout(render, 0); }   // 別蓋掉原本的預設
        const clean = { name: cur.name, stages: cur.stages.map((s) => { const o = { ...s }; delete o._k; return o; }) };
        W("recipe").value = JSON.stringify(clean);
        if (W("preset").value !== CUSTOM) W("preset").value = CUSTOM;
        node.setDirtyCanvas(true, true);
        if (live) { clearTimeout(timer); timer = setTimeout(() => app.queuePrompt(0, 1), 700); }
      };
      const add = (k) => {
        const last = cur.stages[cur.stages.length - 1];
        if (k === "base") cur.stages.push({ model: "base", scheduler: "simple", steps: 25, range: [0, 5] });
        if (k === "fast") cur.stages.push({ model: "fast", sigmas: [0.667, 0.4, 0], lora: 0.93 });
        if (k === "refine") cur.stages.push({ model: "base", sigmas: [0.25, 0.16, 0.08, 0], noise: true, cfg: 3 });
        render(); push();
      };
      const field = (s, key, type, ph) => {
        const i = document.createElement("input"); i.type = type; if (ph) i.placeholder = ph;
        if (type === "number") { i.step = key === "styles" || key === "lora" ? "0.05" : (key === "cfg" ? "0.5" : "1"); }
        const v = s[key]; i.value = Array.isArray(v) ? v.join(",") : (v ?? "");
        if (type === "checkbox") i.checked = !!v;
        i.onchange = () => {
          if (type === "checkbox") s[key] = i.checked;
          else if (key === "sigmas") s.sigmas = i.value.split(",").map(Number).filter((x) => !isNaN(x));
          else if (i.value === "") delete s[key]; else s[key] = Number(i.value);
          render(); push();
        };
        return i;
      };
      const render = () => {
        el.innerHTML = "";
        const head = document.createElement("div"); head.className = "ms-row";
        const nm = document.createElement("input"); nm.value = cur.name || ""; nm.style.width = "180px";
        nm.onchange = () => { cur.name = nm.value; push(); };
        head.append(T.recipe + " ", nm);
        const t = estimate(cur.stages);
        const info = document.createElement("span"); info.className = "ms-note";
        info.textContent = T.est(t.toFixed(1), (COST.fixed + COST.stage * 2 + 30 * COST.base).toFixed(1));
        head.append(info); el.append(head);
        // 流程條
        const bar = document.createElement("div"); bar.className = "ms-bar";
        const tot = cur.stages.reduce((a, s) => a + evals(s) * (s.model === "fast" ? COST.fast : COST.base), 0) || 1;
        cur.stages.forEach((s, i) => {
          const d = document.createElement("div"); const k = kind(s);
          d.style.width = (100 * evals(s) * (s.model === "fast" ? COST.fast : COST.base) / tot) + "%"; d.style.background = COLORS[k];
          d.textContent = `${k} ${nsteps(s)}`; d.title = JSON.stringify(s); bar.append(d);
        });
        el.append(bar);
        // 表格
        const tb = document.createElement("table");
        tb.innerHTML = `<tr><th></th><th></th><th>${T.model}</th><th>${T.sched}</th><th>LoRA</th><th>cfg</th><th>sampler</th><th>styles</th><th>${T.noise}</th></tr>`;
        cur.stages.forEach((s, i) => {
          const tr = document.createElement("tr");
          const td = (...c) => { const x = document.createElement("td"); x.append(...c); tr.append(x); return x; };
          const ops = document.createElement("span");
          const btn = (txt, fn) => { const b = document.createElement("button"); b.textContent = txt; b.onclick = fn; ops.append(b); };
          btn("↑", () => { if (i > 0) { [cur.stages[i - 1], cur.stages[i]] = [cur.stages[i], cur.stages[i - 1]]; render(); push(); } });
          btn("↓", () => { if (i < cur.stages.length - 1) { [cur.stages[i + 1], cur.stages[i]] = [cur.stages[i], cur.stages[i + 1]]; render(); push(); } });
          btn("✕", () => { cur.stages.splice(i, 1); render(); push(); });
          td(ops);
          const dot = document.createElement("span"); dot.textContent = "■"; dot.style.color = COLORS[kind(s)]; td(dot);
          const m = document.createElement("select"); ["base", "fast"].forEach((o) => m.add(new Option(o, o, false, s.model === o)));
          m.onchange = () => { s.model = m.value; if (m.value === "base") delete s.lora; else s.lora = s.lora ?? 0.93; render(); push(); }; td(m);
          if (s.scheduler) {
            const box = document.createElement("span");
            const sc = document.createElement("input"); sc.value = s.scheduler; sc.style.width = "46px"; sc.onchange = () => { s.scheduler = sc.value; push(); };
            box.append(sc, "/", field(s, "steps", "number"));
            const a = document.createElement("input"); a.type = "number"; a.value = s.range[0]; a.onchange = () => { s.range[0] = +a.value; render(); push(); };
            const b = document.createElement("input"); b.type = "number"; b.value = s.range[1]; b.onchange = () => { s.range[1] = +b.value; render(); push(); };
            box.append(" ", a, "–", b); td(box);
          } else { const f = field(s, "sigmas", "text", T.sigph); f.className = "sig"; td(f); }
          if (s.model === "fast") td(field(s, "lora", "number")); else td("");
          td(field(s, "cfg", "number", "1"));
          const sp = document.createElement("select"); SAMPLERS.forEach((o) => sp.add(new Option(o, o, false, (s.sampler || "euler") === o)));
          sp.onchange = () => { if (sp.value === "euler") delete s.sampler; else s.sampler = sp.value; push(); }; td(sp);
          td(field(s, "styles", "number", "1"));
          if (!s.scheduler) td(field(s, "noise", "checkbox")); else td("");
          tb.append(tr);
        });
        el.append(tb);
        const row = document.createElement("div"); row.className = "ms-row";
        const b = (txt, fn) => { const x = document.createElement("button"); x.textContent = txt; x.onclick = fn; row.append(x); };
        b(T.addBase, () => add("base")); b(T.addFast, () => add("fast")); b(T.addRefine, () => add("refine"));
        b(T.save, async () => {
          const name = prompt(T.askName, cur.name); if (!name) return; cur.name = name;
          const res = await api.fetchApi("/multistage/recipes", { method: "POST", body: JSON.stringify({ recipe: { name, stages: cur.stages } }) });
          msg.textContent = res.ok ? T.saved(name) : T.saveFail + (await res.text());
          await load();
        });
        b(T.setDef, async () => {
          const res = await api.fetchApi("/multistage/recipes", { method: "POST", body: JSON.stringify({ recipe: { name: cur.name, stages: cur.stages }, default: true }) });
          msg.textContent = res.ok ? T.savedDef(cur.name) : T.fail + (await res.text()); await load();
        });
        // 合併:從配方庫挑一份接進來
        const mrow = document.createElement("div"); mrow.className = "ms-row";
        const msel = document.createElement("select"); msel.style.maxWidth = "220px";
        lib.recipes.forEach((x) => msel.add(new Option(x.name, x.name)));
        const mmode = document.createElement("select"); mmode.style.maxWidth = "150px";
        [["all", T.mAll], ["last", T.mLast], ["replace", T.mReplace]].forEach(([v, t]) => mmode.add(new Option(t, v)));
        const mb = document.createElement("button"); mb.textContent = T.mBtn;
        mb.onclick = () => {
          const src = lib.recipes.find((x) => x.name === msel.value); if (!src) return;
          let add = JSON.parse(JSON.stringify(src.stages));
          if (mmode.value !== "all") add = add.slice(-1);
          if (mmode.value === "replace") cur.stages.pop();
          const first = add[0];
          const warn = cur.stages.length && first && first.scheduler && first.range && first.range[0] === 0
            ? T.mWarn : "";
          cur.stages.push(...add);
          const tag = src.name.split(" · ")[0];
          if (!cur.name.includes("+ " + tag)) cur.name = `${cur.name.replace(T.edited, "")} + ${tag}`;
          render(); push(); if (warn) { msg2.textContent = warn; }
        };
        mrow.append(T.merge, msel, mmode, mb); el.append(mrow);
        const msg2 = document.createElement("div"); msg2.className = "ms-err"; el.append(msg2);
        const lv = document.createElement("label"); const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = live;
        cb.onchange = () => { live = cb.checked; if (live) app.queuePrompt(0, 1); }; lv.append(cb, T.live); row.append(lv);
        el.append(row);
        const msg = document.createElement("div"); msg.className = "ms-note"; el.append(msg);
        if (!cur.stages.length) { const e = document.createElement("div"); e.className = "ms-err"; e.textContent = T.empty; el.append(e); }
      };

      node.addDOMWidget("recipe_editor", "ms_editor", el, { serialize: false, getMinHeight: () => 230 });
      const pw = W("preset"); const cbk = pw.callback;
      pw.callback = function () { const r2 = cbk?.apply(this, arguments); fromWidgets(); render(); return r2; };
      load().then(() => { fromWidgets(); render(); });
      node.setSize([Math.max(node.size[0], 720), Math.max(node.size[1], 560)]);
      return r;
    };
  },
});
