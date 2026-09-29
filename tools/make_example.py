"""產生範例 workflow:以 ComfyUI 官方「Qwen Image 2.1 文生圖」範本(comfyui_workflow_templates)為底,
把子圖攤平,KSampler 換成 Multi-Stage Sampler,掛 Viggle turbo LoRA、預設「Balanced」配方。
槽位/連線/widget 順序都照 ComfyUI 的 object_info,不手寫。
用法: python tools/make_example.py [comfy host:port] [輸出檔]"""
import glob, json, os, sys, urllib.request
COMFY = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:8188"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "examples", "qwen_image_2_1_t2i_multistage.json")
info = json.load(urllib.request.urlopen("http://%s/object_info" % COMFY))
import comfyui_workflow_templates_json as T                                  # ComfyUI 內建的官方範本包
tpl = json.load(open(glob.glob(os.path.join(os.path.dirname(T.__file__), "templates", "image_qwen_image_2_1_t2i.json"))[0]))
sub = tpl["definitions"]["subgraphs"][0]
tnode = {n["type"]: n for n in sub["nodes"]}
outer = {n["type"]: n for n in tpl["nodes"]}
MODELS = {k: tnode[k]["properties"]["models"] for k in ("UNETLoader", "CLIPLoader", "VAELoader")}
PROMPT = [n for n in tpl["nodes"] if n["type"] not in ("ResolutionSelector", "SaveImageAdvanced", "MarkdownNote")][0]["widgets_values"][0]
VIGGLE = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors"
VIGGLE_MODEL = [{"name": VIGGLE, "url": "https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/resolve/main/" + VIGGLE, "directory": "loras"}]
WIDGET_TYPES = ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO")
def is_widget(sp): return isinstance(sp[0], list) or sp[0] in WIDGET_TYPES

# (id, 類別, 位置, 大小, widget 值, 連入 {輸入: (來源 id, 槽)}, mode, 標題, models)
N = [
 (1, "UNETLoader", (40, 40), (380, 110), {"unet_name": tnode["UNETLoader"]["widgets_values"][0], "weight_dtype": "default"}, {}, 0, None, MODELS["UNETLoader"]),
 (2, "LoraLoaderModelOnly", (40, 190), (380, 110), {"lora_name": "your_style_lora.safetensors", "strength_model": 1.0}, {"model": (1, 0)}, 4,
     "Style / character LoRA (optional: pick yours, un-bypass)", None),
 (3, "CLIPLoader", (40, 340), (380, 130), {"clip_name": tnode["CLIPLoader"]["widgets_values"][0], "type": "qwen_image", "device": "default"}, {}, 0, None, MODELS["CLIPLoader"]),
 (4, "VAELoader", (40, 510), (380, 80), {"vae_name": tnode["VAELoader"]["widgets_values"][0]}, {}, 0, None, MODELS["VAELoader"]),
 (5, "TextEncodeQwenImage21", (460, 40), (460, 330), {"prompt": PROMPT, "negative_prompt": "", "resolution": 1024}, {"clip": (3, 0), "vae": (4, 0)}, 0, None, None),
 (6, "ResolutionSelector", (460, 410), (300, 190), dict(zip(["aspect_ratio", "megapixels", "multiple_of"], outer["ResolutionSelector"]["widgets_values"])), {}, 0, None, None),
 (7, "EmptyLatentImage", (460, 640), (300, 110), {"width": 1024, "height": 1024, "batch_size": 1}, {"width": (6, 0), "height": (6, 1)}, 0, None, None),
 (8, "MultiStageSampler", (960, 40), (760, 600), {"seed": 0, "fast_lora": VIGGLE, "preset": "Balanced · copy 5 + fast 3", "recipe": ""},
     {"model": (2, 0), "positive": (5, 0), "negative": (5, 1), "latent_image": (7, 0)}, 0, None, VIGGLE_MODEL),
 (9, "VAEDecode", (1760, 40), (220, 50), {}, {"samples": (8, 0), "vae": (4, 0)}, 0, None, None),
 (10, "SaveImageAdvanced", (2020, 40), (850, 610), dict(zip(["filename_prefix", "format", "bit_depth", "color_space"], ["Qwen_image_2.1_multistage"] + outer["SaveImageAdvanced"]["widgets_values"][1:])),
     {"images": (9, 0)}, 0, None, None),
 (11, "PreviewAny", (1760, 160), (240, 160), {}, {"source": (8, 1)}, 0, "Stage log (steps / σ / seconds)", None),
]
nodes, links, lid, out_links = [], [], 0, {}
for nid, cls, pos, size, wv, ins, mode, title, models in N:
    d = info[cls]
    req, opt = d["input"].get("required", {}), d["input"].get("optional", {})
    order = d.get("input_order", {})
    names = order.get("required", list(req)) + order.get("optional", list(opt))
    spec = dict(req, **opt)
    inputs, widgets = [], []
    for n in names:
        sp = spec[n]
        if is_widget(sp):
            v = wv.get(n, sp[1].get("default") if len(sp) > 1 and isinstance(sp[1], dict) else None)
            if v is None and isinstance(sp[0], list): v = sp[0][0]
            widgets.append(v)
            if len(sp) > 1 and isinstance(sp[1], dict) and sp[1].get("control_after_generate"):
                widgets.append("fixed")
            inputs.append({"name": n, "type": sp[0] if isinstance(sp[0], str) else "COMBO", "widget": {"name": n}, "link": None})
        elif sp[0] != "COMFY_AUTOGROW_V3":
            inputs.append({"name": n, "type": sp[0], "link": None})
    for n, (src, slot) in ins.items():
        lid += 1
        stype = info[[x for x in N if x[0] == src][0][1]]["output"][slot]
        links.append([lid, src, slot, nid, [i["name"] for i in inputs].index(n), stype])
        [i for i in inputs if i["name"] == n][0]["link"] = lid
        out_links.setdefault((src, slot), []).append(lid)
    outputs = [{"name": nm, "type": t, "links": [], "slot_index": k} for k, (t, nm) in enumerate(zip(d["output"], d.get("output_name", d["output"])))]
    props = {"Node name for S&R": cls}
    if models: props["models"] = models
    node = {"id": nid, "type": cls, "pos": list(pos), "size": list(size), "flags": {}, "order": nid - 1, "mode": mode,
            "inputs": inputs, "outputs": outputs, "properties": props, "widgets_values": widgets}
    if title: node["title"] = title
    nodes.append(node)
for n in nodes:
    for o in n["outputs"]:
        o["links"] = out_links.get((n["id"], o["slot_index"]), [])
model_links = [x for x in tpl["nodes"] if x["type"] == "MarkdownNote" and "Model Links" in x["widgets_values"][0]][0]["widgets_values"][0]
model_links = model_links.split("## Model Links", 1)[1].split("## Report Issue")[0]
note = ("# Qwen Image 2.1 + Multi-Stage Sampler\n\n"
        "Based on the official *Qwen Image 2.1 text to image* template; the KSampler is replaced by **Multi-Stage Sampler**: "
        "the base model draws the first steps of the 25-step schedule (same composition as plain 25 steps), "
        "then a fast LoRA finishes in 3 steps — about 2.5× faster.\n\n"
        "## Recipes (preset)\n- **Fast** · copy 4 + fast 3\n- **Balanced** · copy 5 + fast 3 (default)\n- **Fine** · copy 6 + fast 3\n"
        "- **Text** · balanced + renoise refine (σ 0.25, cfg 3) — for lettering\n- **Reference 25 steps** (cfg 1 ×20, cfg 3 ×5)\n\n"
        "Edit stages in the editor under the node, or pick `custom` and paste a JSON recipe. Recipes: `ComfyUI/user/multistage_recipes.json`.\n\n"
        "## Fast LoRA\n- [Viggle/Qwen-Image-2.1-viggle-turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) → `models/loras/" + VIGGLE + "`\n"
        "- Also install its `comfyui/viggle_turbo.py` node: the LoRA is then applied unmerged (exact). Without it the sampler falls back to ComfyUI's standard (merged) LoRA.\n\n"
        "## Style / character LoRAs\nPut them in front of the model as usual (all stages use them), or feed a `LORA_STACK` and set a per-stage `styles` multiplier.\n\n"
        "## Model Links" + model_links)
nodes.append({"id": 12, "type": "MarkdownNote", "pos": [-480, 40], "size": [480, 900], "flags": {}, "order": 11, "mode": 0, "inputs": [], "outputs": [],
              "properties": {}, "widgets_values": [note], "color": "#432", "bgcolor": "#653"})
wf = {"last_node_id": 12, "last_link_id": lid, "nodes": nodes, "links": links, "groups": [], "config": {}, "extra": {}, "version": 0.4}
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
json.dump(wf, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", OUT, len(nodes), "nodes", lid, "links")
