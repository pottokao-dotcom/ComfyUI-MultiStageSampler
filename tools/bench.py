"""Multi-Stage Sampler bench:標準樣本 → 參考圖 → 候選配方 → 指標 + 成本回歸 → 自動挑配方。
用法(在 ComfyUI 的 venv 跑,ComfyUI 要開著):
  python tools/bench.py ref   bench/qi21_v1.json                  # 產生/快取參考圖
  python tools/bench.py run   bench/qi21_v1.json cands.json         # 跑候選配方(JSON 配方陣列)
  python tools/bench.py grid  bench/qi21_v1.json grid.json          # 由格點規格展開候選再跑
  python tools/bench.py fit   bench/qi21_v1.json [out_recipes.json] # 成本回歸 + 前緣 + 依 tiers 挑配方
  python tools/bench.py upgrade bench/qi21_v2.json                  # 樣本加了以後,舊配方只補跑新題目
結果在 bench/<name>/results.jsonl(一個配方一行,含每題指標與時間),同配方(依 stages 內容)不重跑。"""
import hashlib, io, itertools, json, os, sys, time, urllib.parse, urllib.request
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)


def rkey(recipe):
    return hashlib.sha1(json.dumps(recipe["stages"], sort_keys=True).encode()).hexdigest()[:12]


class Bench:
    def __init__(self, spec_path):
        self.spec = json.load(open(spec_path, encoding="utf-8"))
        self.dir = os.path.join(os.path.dirname(os.path.abspath(spec_path)), self.spec["name"])
        os.makedirs(os.path.join(self.dir, "ref"), exist_ok=True); os.makedirs(os.path.join(self.dir, "img"), exist_ok=True)
        self.res_path = os.path.join(self.dir, "results.jsonl")

    def comfy(self, path, data=None):
        req = urllib.request.Request("http://%s%s" % (self.spec["comfy"], path),
                                     json.dumps(data).encode() if data is not None else None, {"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=60).read()

    def render(self, sample, recipe, save_to):
        def fill(v):
            if isinstance(v, str):
                return {"{TEXT}": sample["text"], "{NEG}": sample.get("negative", ""), "{W}": sample["width"], "{H}": sample["height"]}.get(v, v)
            if isinstance(v, dict):
                return {k: fill(x) for k, x in v.items()}
            return v
        g = fill(json.loads(json.dumps(self.spec["graph"])))
        g["ms"] = {"class_type": "MultiStageSampler", "inputs": {
            "model": ["model", 0], "positive": self.spec["positive"], "negative": self.spec["negative"], "latent_image": ["lat", 0],
            "seed": sample["seed"], "fast_lora": self.spec.get("fast_lora", "none"), "preset": "custom",
            "recipe": json.dumps(recipe, ensure_ascii=False)}}
        g["info"] = {"class_type": "PreviewAny", "inputs": {"source": ["ms", 1]}}
        g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["ms", 0], "vae": ["vae", 0]}}
        g["save"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "msbench/x", "images": ["dec", 0]}}
        t = time.time()
        pid = json.loads(self.comfy("/prompt", {"prompt": g, "client_id": "msbench"}))["prompt_id"]
        while True:
            h = json.loads(self.comfy("/history/" + pid))
            if pid in h and h[pid].get("status", {}).get("completed"):
                break
            if pid in h and h[pid].get("status", {}).get("status_str") == "error":
                raise RuntimeError(json.dumps(h[pid]["status"]["messages"])[-800:])
            time.sleep(0.2)
        dt = time.time() - t
        out = h[pid]["outputs"]; im = out["save"]["images"][0]
        raw = self.comfy("/view?" + urllib.parse.urlencode({"filename": im["filename"], "subfolder": im["subfolder"], "type": im["type"]}))
        Image.open(io.BytesIO(raw)).save(save_to)
        return dt, (out["info"].get("text") or [""])[0]

    def ref(self):
        for s in self.spec["samples"]:
            p = os.path.join(self.dir, "ref", s["id"] + ".png")
            if not os.path.exists(p):
                dt, _ = self.render(s, self.spec["reference"], p); print("ref", s["id"], "%.1fs" % dt, flush=True)
        return self

    def done(self):
        if not os.path.exists(self.res_path):
            return {}
        return {json.loads(l)["key"]: json.loads(l) for l in open(self.res_path, encoding="utf-8")}

    def run(self, recipes):
        """樣本可以後加:同一配方已經跑過的題目不重跑,只補新題目,再重算平均。"""
        import metrics as M
        self.ref(); have = self.done()
        ids = [s["id"] for s in self.spec["samples"]]
        for rc in recipes:
            k = rkey(rc)
            old = {r["id"]: r for r in have.get(k, {}).get("rows", []) if r["id"] in ids}
            if len(old) == len(ids):
                continue
            rows = []
            for s in self.spec["samples"]:
                if s["id"] in old:
                    rows.append(old[s["id"]]); continue
                p = os.path.join(self.dir, "img", "%s_%s.png" % (k, s["id"]))
                dt, info = self.render(s, rc, p)
                rows.append(dict(M.compare(os.path.join(self.dir, "ref", s["id"] + ".png"), p), id=s["id"], time=dt))
            have[k] = self.summarize(k, rc, rows)
            self.save(have)
            r = have[k]
            print("%-34s %5.1fs LPIPS %.3f worst %.3f dinoP %.3f sharp %.2f" % (rc["name"][:34], r["time"], r["lpips"], r["worst"],
                                                                              r["dinop"], r["sharp"]), flush=True)

    @staticmethod
    def summarize(k, rc, rows):
        L = [r["lpips"] for r in rows]
        return {"key": k, "recipe": rc, "rows": rows, "time": float(np.median([r["time"] for r in rows])),
                "lpips": float(np.mean(L)), "worst": float(max(L)),
                "dinop": float(np.mean([r["dinop"] for r in rows])), "sharp": float(np.mean([r["sharp"] for r in rows])),
                "mae64": float(np.mean([r["mae64"] for r in rows]))}

    def save(self, have):
        tmp = self.res_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in have.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, self.res_path)

    def upgrade(self):
        """spec 樣本變多後:把所有已評估過的配方補跑新題目。"""
        self.run([r["recipe"] for r in self.done().values()])


def features(rc):
    nb = nf = 0
    for st in rc["stages"]:
        n = (st["range"][1] - st["range"][0]) if "scheduler" in st else len(st["sigmas"])
        n *= 2 if st.get("cfg", 1) > 1 else 1                 # CFG 一步要算兩次
        if st["model"] == "base": nb += n
        else: nf += n
    return [1.0, nb, nf, len(rc["stages"])]


def grid(spec):
    """{"copy_steps":[3,4,…], "fast_sigmas":[[0.667,0.4,0],…], "lora":[0.95,…], "scheduler":"simple", "steps":25}"""
    out = []
    for n, sg, lo in itertools.product(spec["copy_steps"], spec["fast_sigmas"], spec["lora"]):
        out.append({"name": "抄%d + fast%d lora%.2f σ%s" % (n, len(sg), lo, ",".join("%g" % x for x in sg)),
                    "stages": [{"model": "base", "scheduler": spec.get("scheduler", "simple"), "steps": spec.get("steps", 25), "range": [0, n]},
                               {"model": "fast", "sigmas": sg, "lora": lo}]})
    return out


def fit(b, out_path=None):
    rs = list(b.done().values())
    X = np.array([features(r["recipe"]) for r in rs]); y = np.array([r["time"] for r in rs])
    c = np.linalg.lstsq(X, y, rcond=None)[0] if len(rs) >= 4 else None
    if c is not None:
        print("成本回歸:固定 %.2fs + base 每步 %.3fs + fast 每步 %.3fs + 每段 %.3fs(最大誤差 %.2fs,n=%d)" % (
            *c, float(np.abs(X @ c - y).max()), len(rs)))
    lo, hi = b.spec.get("sharp_range", [0, 99])
    ok = sorted([r for r in rs if lo <= r["sharp"] <= hi], key=lambda r: r["time"])
    front, best = [], 9
    for r in ok:                                             # 時間由小到大,LPIPS 創新低的才進前緣
        if r["lpips"] < best:
            front.append(r); best = r["lpips"]
    print("\n前緣(銳度 %.2f–%.2f):" % (lo, hi))
    for r in front:
        print("  %5.1fs  LPIPS %.3f  worst %.3f  sharp %.2f  %s" % (r["time"], r["lpips"], r["worst"], r["sharp"], r["recipe"]["name"]))
    picks = []
    for t in b.spec.get("tiers", []):
        c2 = [r for r in front if r["lpips"] <= t["max_lpips"] and r["worst"] <= t.get("max_worst", 9)]
        if c2:
            r = c2[0]; rc = dict(r["recipe"], name="%s · %s" % (t["name"], r["recipe"]["name"]),
                                 note="bench %s:%.1fs,LPIPS %.3f(最差 %.3f),銳度 %.2f" % (b.spec["name"], r["time"], r["lpips"], r["worst"], r["sharp"]))
            picks.append(rc); print("→ %s:%s" % (t["name"], rc["note"]))
        else:
            print("→ %s:沒有配方達標" % t["name"])
    if out_path and picks:
        json.dump({"default": picks[0]["name"], "recipes": picks + [dict(b.spec["reference"], note="基準")]},
                  open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("寫入", out_path)


if __name__ == "__main__":
    cmd, spec = sys.argv[1], sys.argv[2]
    b = Bench(spec)
    if cmd == "ref":
        b.ref()
    elif cmd == "run":
        b.run(json.load(open(sys.argv[3], encoding="utf-8")))
    elif cmd == "grid":
        b.run(grid(json.load(open(sys.argv[3], encoding="utf-8"))))
    elif cmd == "upgrade":
        b.upgrade()
    elif cmd == "fit":
        fit(b, sys.argv[3] if len(sys.argv) > 3 else None)
