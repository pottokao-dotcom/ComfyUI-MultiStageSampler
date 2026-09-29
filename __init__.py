"""Multi-Stage Sampler:一個節點跑「多段取樣」—— 每段選原模型或 fast LoRA、步數/σ、LoRA 強度、CFG、sampler。
省掉 SplitSigmas / 多個 SamplerCustomAdvanced / Guider 的拉線。風格、人物 LoRA 照常:
  (a) 模型先經過你自己的 LoRA Loader 再接進來;或 (b) 接 LORA_STACK(rgthree / Efficiency 等 stacker 格式),每段可用 "styles" 設倍率(0 = 不掛)。
配方(JSON):ComfyUI/user/multistage_recipes.json(第一次自動從 recipes_default.json 複製)。"""
import json, os, shutil, time
import torch
import comfy.samplers, comfy.sample, comfy.sd, comfy.utils, comfy.model_management
import folder_paths, latent_preview
from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_RandomNoise, Noise_EmptyNoise

HERE = os.path.dirname(os.path.abspath(__file__))
CUSTOM = "custom"
CUSTOM_LEGACY = ("custom(用下面的 recipe)",)          # 舊版 workflow 存的值


def presets_path():
    p = os.path.join(folder_paths.get_user_directory(), "multistage_recipes.json")
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        shutil.copy(os.path.join(HERE, "recipes_default.json"), p)
    return p


def load_presets():
    """multistage_recipes.json = {"default": 名稱, "recipes": [配方, ...]}。回傳 (名稱→配方, 預設名稱)。"""
    d = json.load(open(presets_path(), encoding="utf-8"))
    out = {r["name"]: r for r in d.get("recipes", [])}
    return out, d.get("default") if d.get("default") in out else (next(iter(out)) if out else None)


def parse(recipe):
    """配方 = {"name":…, "stages":[…]};每段:
       {"model":"base"|"fast", "scheduler":"simple","steps":25,"range":[a,b]}  照抄該排程第 a~b 步
    或 {"model":…, "sigmas":[…]}  從上一段終點接著走到這些 σ
       選填 "lora"(fast 段強度,預設 1)、"cfg"(預設 1)、"sampler"(預設 euler)、"styles"(LORA_STACK 強度倍率,預設 1;0/false = 不掛)、
       "noise": true(只配 sigmas:加新雜訊到 sigmas[0] 再走,用在收尾 refine)。"""
    if isinstance(recipe, str):
        recipe = json.loads(recipe)
    stages = recipe["stages"] if isinstance(recipe, dict) else recipe
    out = []
    for n, st in enumerate(stages, 1):
        st = dict({"lora": 1.0, "cfg": 1.0, "sampler": "euler", "styles": 1.0}, **st)
        st["styles"] = float(st["styles"])          # true/false 相容 → 1/0;數字 = LORA_STACK 強度倍率
        if st.get("model") not in ("base", "fast"):
            raise ValueError("stage %d: model must be base or fast" % n)
        if ("scheduler" in st) == ("sigmas" in st):
            raise ValueError("stage %d: give either scheduler+steps(+range) or sigmas" % n)
        if "scheduler" in st:
            st.setdefault("range", [0, st["steps"]])
        if st["sampler"] not in comfy.samplers.SAMPLER_NAMES:
            raise ValueError("stage %d: unknown sampler %s" % (n, st["sampler"]))
        st["line"] = json.dumps({k: v for k, v in st.items() if k != "line"}, ensure_ascii=False)
        out.append(st)
    if not out:
        raise ValueError("recipe has no stages")
    return out


def with_fast_lora(model, name, strength):
    """Viggle turbo 這類要「不合併」才準的 LoRA:有裝 ViggleTurboLora 節點就用它(執行時 Wx+BAx);否則用 ComfyUI 標準合併。"""
    import nodes
    V = nodes.NODE_CLASS_MAPPINGS.get("ViggleTurboLora")
    if V is not None and "viggle" in name.lower():
        return V().load(model, name, strength)[0], "unmerged"
    lora = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", name), safe_load=True)
    return comfy.sd.load_lora_for_models(model, None, lora, strength, 0)[0], "merged"


def with_stack(model, stack, scale=1.0):
    """scale = 這段的 styles 倍率(每個 LoRA 的 model 強度 × scale)。"""
    for item in stack or []:
        name, sm = item[0], float(item[1]) * scale
        if not name or name == "None" or sm == 0:
            continue
        lora = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", name), safe_load=True)
        model = comfy.sd.load_lora_for_models(model, None, lora, sm, 0)[0]
    return model


class MultiStageSampler:
    @classmethod
    def INPUT_TYPES(cls):
        names, default = load_presets()
        return {"required": {
                    "model": ("MODEL", {"tooltip": "Base model (LoRA loaders can go in front of it)"}),
                    "positive": ("CONDITIONING",),
                    "latent_image": ("LATENT",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                    "fast_lora": (["none"] + folder_paths.get_filename_list("loras"), {"tooltip": "Fast / turbo LoRA, applied only in fast stages"}),
                    "preset": (list(names) + [CUSTOM], {"default": default or CUSTOM,
                               "tooltip": "Recipes live in ComfyUI/user/multistage_recipes.json (\"default\" = preselected)"}),
                    "recipe": ("STRING", {"multiline": True,
                               "default": json.dumps(names[default], ensure_ascii=False, indent=1) if default else "",
                               "tooltip": "Used when preset is custom: paste a JSON recipe {\"name\":…,\"stages\":[…]}"}),
                },
                "optional": {
                    "negative": ("CONDITIONING", {"tooltip": "Only needed by stages with cfg > 1"}),
                    "lora_stack": ("LORA_STACK", {"tooltip": "Style / character LoRAs [(name, model strength, clip strength)]; per-stage \"styles\" multiplier (0 = off)"}),
                    "fast_model": ("MODEL", {"tooltip": "Separate fast/turbo model (e.g. Z-Image Turbo): fast stages use it instead of model; fast_lora is optional then"}),
                }}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "info")
    FUNCTION = "run"
    CATEGORY = "sampling/custom_sampling"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return open(presets_path(), encoding="utf-8").read() if kw.get("preset") != CUSTOM else ""

    def run(self, model, positive, latent_image, seed, fast_lora, preset, recipe, negative=None, lora_stack=None, fast_model=None):
        if preset in CUSTOM_LEGACY:
            preset = CUSTOM
        if preset != CUSTOM:
            names, _ = load_presets()
            if preset not in names:
                raise ValueError("preset \"%s\" not found in %s" % (preset, presets_path()))
            recipe = names[preset]
        stages = parse(recipe)
        if any(s["model"] == "fast" for s in stages) and fast_lora == "none" and fast_model is None:
            raise ValueError("recipe has fast stages: pick a fast_lora or connect fast_model")
        if any(s["cfg"] > 1 for s in stages) and negative is None:
            raise ValueError("recipe has a stage with cfg > 1: connect negative")

        styled = {}
        cache = {}

        def model_for(st):
            key = (st["model"], st["lora"] if st["model"] == "fast" else None, st["styles"])
            if key not in cache:
                sc = st["styles"]
                src = fast_model if (st["model"] == "fast" and fast_model is not None) else model
                if (id(src), sc) not in styled:
                    styled[(id(src), sc)] = with_stack(src, lora_stack, sc) if (lora_stack and sc != 0) else src
                m = styled[(id(src), sc)]
                if st["model"] == "fast" and fast_lora != "none":
                    cache[key] = with_fast_lora(m, fast_lora, st["lora"])
                else:
                    cache[key] = (m, "fast_model" if src is fast_model and st["model"] == "fast" else "")
            return cache[key]

        latent = latent_image.copy()
        x = comfy.sample.fix_empty_latent_channels(model, latent["samples"], latent.get("downscale_ratio_spacial"),
                                                   latent.get("downscale_ratio_temporal"))
        cur = None                  # 目前的 σ
        info = []
        t_all = time.time()
        for i, st in enumerate(stages):
            m, how = model_for(st)
            if "scheduler" in st:
                full = comfy.samplers.calculate_sigmas(m.get_model_object("model_sampling"), st["scheduler"], st["steps"]).cpu()
                a, b = st["range"]
                sig = full[a:b + 1]
            elif st.get("noise"):                          # refine:加新雜訊到 sigmas[0] 再走(img2img 式,x=σ·ε+(1−σ)·x0)
                sig = torch.tensor(st["sigmas"], dtype=torch.float32)
            else:
                if cur is None:
                    raise ValueError("the first stage needs a schedule (or noise: true); plain sigmas have no starting point")
                sig = torch.tensor([cur] + st["sigmas"], dtype=torch.float32)
            if len(sig) < 2:
                raise ValueError("stage has no steps: %s" % st["line"])
            if cur is not None and abs(float(sig[0]) - cur) > 1e-3 and not st.get("noise"):
                info.append("⚠ stage %d starts at σ %.4f but the previous stage ended at %.4f" % (i + 1, float(sig[0]), cur))
            if st["cfg"] > 1:
                g = comfy.samplers.CFGGuider(m); g.set_conds(positive, negative); g.set_cfg(st["cfg"])
            else:
                g = Guider_Basic(m); g.set_conds(positive)
            noise = Noise_RandomNoise(seed) if i == 0 else (Noise_RandomNoise(seed + 1000 + i) if st.get("noise") else Noise_EmptyNoise())
            t0 = time.time()
            cb = latent_preview.prepare_callback(m, len(sig) - 1, {})
            x = g.sample(noise.generate_noise(dict(latent, samples=x)), x, comfy.samplers.sampler_object(st["sampler"]), sig,
                         denoise_mask=latent.get("noise_mask"), callback=cb,
                         disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
            x = x.to(comfy.model_management.intermediate_device())
            cur = float(sig[-1])
            info.append("%d. %s %d steps σ %.3f→%.3f%s%s%s  %.2fs" % (
                i + 1, st["model"], len(sig) - 1, float(sig[0]), cur,
                ((" lora %.2f(%s)" % (st["lora"], how)) if how not in ("", "fast_model") else (" (fast_model)" if how == "fast_model" else "")) if st["model"] == "fast" else "",
                ((" cfg %.1f" % st["cfg"]) if st["cfg"] > 1 else "") + (" +noise" if st.get("noise") else ""),
                ("" if not lora_stack else (" styles off" if st["styles"] == 0 else ("" if st["styles"] == 1 else " styles ×%g" % st["styles"]))),
                time.time() - t0))
        if cur is not None and cur > 1e-4:
            info.append("⚠ last σ %.4f is not 0: the image keeps residual noise" % cur)
        info.append("total %.2fs (%s)" % (time.time() - t_all, preset))
        out = latent.copy(); out.pop("downscale_ratio_spacial", None); out.pop("downscale_ratio_temporal", None)
        out["samples"] = x
        return (out, "\n".join(info))


class MultiStageLoraStack:
    """LORA_STACK 產生器,格式同 Efficiency「LoRA Stacker」/ Comfyroll「CR LoRA Stack」:[(名稱, model 強度, clip 強度), …],可串接。"""
    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        req = {}
        for i in (1, 2, 3):
            req["lora_%d" % i] = (loras,)
            req["strength_%d" % i] = ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05})
        return {"required": req, "optional": {"lora_stack": ("LORA_STACK",)}}

    RETURN_TYPES = ("LORA_STACK",)
    FUNCTION = "stack"
    CATEGORY = "loaders"

    def stack(self, lora_stack=None, **kw):
        out = list(lora_stack or [])
        for i in (1, 2, 3):
            name, st = kw["lora_%d" % i], kw["strength_%d" % i]
            if name != "None" and st != 0:
                out.append((name, st, st))
        return (out,)


# ---- 編輯器用的 API(web/multistage.js)----
try:
    from server import PromptServer
    from aiohttp import web as _web
    _routes = PromptServer.instance.routes

    @_routes.get("/multistage/recipes")
    async def _get_recipes(request):
        return _web.json_response(json.load(open(presets_path(), encoding="utf-8")))

    @_routes.post("/multistage/recipes")
    async def _save_recipe(request):
        """{recipe: {name, stages, note?}, default?: bool, delete?: name}"""
        body = await request.json()
        d = json.load(open(presets_path(), encoding="utf-8"))
        if body.get("delete"):
            d["recipes"] = [r for r in d["recipes"] if r["name"] != body["delete"]]
        else:
            rc = body["recipe"]
            parse(rc)                                  # 格式錯就擋下來
            d["recipes"] = [r for r in d["recipes"] if r["name"] != rc["name"]] + [rc]
            if body.get("default"):
                d["default"] = rc["name"]
        tmp = presets_path() + ".tmp"
        json.dump(d, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        os.replace(tmp, presets_path())
        return _web.json_response(d)
except Exception as e:                                  # 沒有 server(例如 bench 單獨 import)就略過
    print("[MultiStageSampler] editor API not loaded:", e)

WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS = {"MultiStageSampler": MultiStageSampler, "MultiStageLoraStack": MultiStageLoraStack}
NODE_DISPLAY_NAME_MAPPINGS = {"MultiStageSampler": "Multi-Stage Sampler",
                              "MultiStageLoraStack": "Multi-Stage LoRA Stack"}
