"""Z-Image base + Turbo 範例:以 ComfyUI 官方 Z-Image(base)與 Z-Image Turbo 範本的模型為底,
base 掛 alibaba-pai Distill 8-step LoRA 0.8 → Multi-Stage Sampler(base 前 2 步 + turbo 後 10 步,res_multistep)。
用法: python tools/make_example_zimage.py [comfy host:port] [輸出檔]"""
import glob, json, os, sys, urllib.request
COMFY = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:8188"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "examples", "z_image_base_turbo_multistage.json")
info = json.load(urllib.request.urlopen("http://%s/object_info" % COMFY))
import comfyui_workflow_templates_json as T
D = os.path.join(os.path.dirname(T.__file__), "templates")
def sub(f): return {n["type"]: n for n in json.load(open(os.path.join(D, f)))["definitions"]["subgraphs"][0]["nodes"]}
base, turbo = sub("image_z_image.json"), sub("image_z_image_turbo.json")
M = lambda n: n["properties"]["models"]
DISTILL = "Z-Image-Fun-Lora-Distill-8-Steps-2603-ComfyUI.safetensors"
DISTILL_M = [{"name": DISTILL, "url": "https://huggingface.co/alibaba-pai/Z-Image-Fun-Lora-Distill/resolve/main/" + DISTILL, "directory": "loras"}]
PROMPT = turbo["CLIPTextEncode"]["widgets_values"][0]
WT = ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO")
def is_widget(sp): return isinstance(sp[0], list) or sp[0] in WT
N = [
 (1, "UNETLoader", (40, 40), (380, 110), {"unet_name": base["UNETLoader"]["widgets_values"][0], "weight_dtype": "default"}, {}, 0, "Z-Image (base)", M(base["UNETLoader"])),
 (2, "LoraLoaderModelOnly", (40, 190), (380, 110), {"lora_name": DISTILL, "strength_model": 0.8}, {"model": (1, 0)}, 0, "Distill 8-step LoRA 0.8 (base only)", DISTILL_M),
 (3, "ModelSamplingAuraFlow", (40, 340), (380, 80), {"shift": 3.0}, {"model": (2, 0)}, 0, None, None),
 (4, "UNETLoader", (40, 460), (380, 110), {"unet_name": turbo["UNETLoader"]["widgets_values"][0], "weight_dtype": "default"}, {}, 0, "Z-Image Turbo", M(turbo["UNETLoader"])),
 (5, "ModelSamplingAuraFlow", (40, 610), (380, 80), {"shift": 3.0}, {"model": (4, 0)}, 0, None, None),
 (6, "CLIPLoader", (40, 730), (380, 130), {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2", "device": "default"}, {}, 0, None, M(turbo["CLIPLoader"])),
 (7, "VAELoader", (40, 900), (380, 80), {"vae_name": "ae.safetensors"}, {}, 0, None, M(turbo["VAELoader"])),
 (8, "CLIPTextEncode", (460, 40), (440, 260), {"text": PROMPT}, {"clip": (6, 0)}, 0, None, None),
 (9, "ConditioningZeroOut", (460, 340), (300, 50), {}, {"conditioning": (8, 0)}, 0, None, None),
 (10, "EmptySD3LatentImage", (460, 430), (300, 110), {"width": 1024, "height": 1024, "batch_size": 1}, {}, 0, None, None),
 (11, "MultiStageSampler", (940, 40), (760, 600), {"seed": 0, "fast_lora": "none", "preset": "Z-Image · base 2 + turbo 10", "recipe": ""},
     {"model": (3, 0), "fast_model": (5, 0), "positive": (8, 0), "negative": (9, 0), "latent_image": (10, 0)}, 0, None, None),
 (12, "VAEDecode", (1740, 40), (220, 50), {}, {"samples": (11, 0), "vae": (7, 0)}, 0, None, None),
 (13, "SaveImage", (2000, 40), (600, 640), {"filename_prefix": "z_image_multistage"}, {"images": (12, 0)}, 0, None, None),
 (14, "PreviewAny", (1740, 160), (240, 160), {}, {"source": (11, 1)}, 0, "Stage log (steps / σ / seconds)", None),
]
nodes, links, lid, out_links = [], [], 0, {}
for nid, cls, pos, size, wv, ins, mode, title, models in N:
    d = info[cls]; req, opt = d["input"].get("required", {}), d["input"].get("optional", {})
    order = d.get("input_order", {}); names = order.get("required", list(req)) + order.get("optional", list(opt)); spec = dict(req, **opt)
    inputs, widgets = [], []
    for n in names:
        sp = spec[n]
        if is_widget(sp):
            v = wv.get(n, sp[1].get("default") if len(sp) > 1 and isinstance(sp[1], dict) else None)
            if v is None and isinstance(sp[0], list): v = sp[0][0]
            widgets.append(v)
            if len(sp) > 1 and isinstance(sp[1], dict) and sp[1].get("control_after_generate"): widgets.append("fixed")
            inputs.append({"name": n, "type": sp[0] if isinstance(sp[0], str) else "COMBO", "widget": {"name": n}, "link": None})
        elif sp[0] != "COMFY_AUTOGROW_V3":
            inputs.append({"name": n, "type": sp[0], "link": None})
    for n, (src, slot) in ins.items():
        lid += 1; stype = info[[x for x in N if x[0] == src][0][1]]["output"][slot]
        links.append([lid, src, slot, nid, [i["name"] for i in inputs].index(n), stype])
        [i for i in inputs if i["name"] == n][0]["link"] = lid; out_links.setdefault((src, slot), []).append(lid)
    outputs = [{"name": nm, "type": t, "links": [], "slot_index": k} for k, (t, nm) in enumerate(zip(d["output"], d.get("output_name", d["output"])))]
    props = {"Node name for S&R": cls}
    if models: props["models"] = models
    node = {"id": nid, "type": cls, "pos": list(pos), "size": list(size), "flags": {}, "order": nid - 1, "mode": mode,
            "inputs": inputs, "outputs": outputs, "properties": props, "widgets_values": widgets}
    if title: node["title"] = title
    nodes.append(node)
for n in nodes:
    for o in n["outputs"]: o["links"] = out_links.get((n["id"], o["slot_index"]), [])
note = ("# Z-Image base + Turbo with Multi-Stage Sampler\n\n"
        "The **base** model (with the Distill 8-step LoRA at 0.8) draws the first 2 of 12 steps and fixes the composition; "
        "**Z-Image Turbo** (connected to `fast_model`) finishes the other 10. res_multistep, simple schedule, shift 3, cfg 1. "
        "Same result as two chained KSamplerAdvanced nodes, in one node.\n\n"
        "## Models\n"
        "- [z_image_bf16.safetensors](https://huggingface.co/Comfy-Org/z_image/resolve/main/split_files/diffusion_models/z_image_bf16.safetensors) → `diffusion_models/`\n"
        "- [z_image_turbo_bf16.safetensors](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors) → `diffusion_models/`\n"
        "- [" + DISTILL + "](" + DISTILL_M[0]["url"] + ") → `loras/`\n"
        "- [qwen_3_4b.safetensors](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors) → `text_encoders/`\n"
        "- [ae.safetensors](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors) → `vae/`\n")
nodes.append({"id": 15, "type": "MarkdownNote", "pos": [-480, 40], "size": [480, 620], "flags": {}, "order": 14, "mode": 0, "inputs": [], "outputs": [],
              "properties": {}, "widgets_values": [note], "color": "#432", "bgcolor": "#653"})
wf = {"last_node_id": 15, "last_link_id": lid, "nodes": nodes, "links": links, "groups": [], "config": {}, "extra": {}, "version": 0.4}
json.dump(wf, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1); print("wrote", OUT, len(nodes), "nodes", lid, "links")
