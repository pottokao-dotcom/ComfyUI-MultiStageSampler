"""跟參考圖(原 25 步)比的多種指標。不改 ComfyUI 環境:lpips 裝在 ~/metrics_pkgs,DINOv2 用 transformers。
指標(全部「越小越像」,除了 sharp):
  mae64  : 縮 64×64 RGB 平均絕對差(0–255)— 構圖/大色塊
  ssim   : 1−SSIM(灰階 512 短邊)— 結構
  lpips  : LPIPS(alex,512 短邊)— 人眼感知(紋理也算)
  dino   : 1−cos(DINOv2-small CLS)— 內容語意;dinop = patch 特徵平均 1−cos — 構圖語意(Viggle 量構圖用 DINOv2 patch)
  sharp  : Laplacian 變異數 ÷ 參考圖的(1.0 = 一樣銳;>1 = 更多高頻,「毛毛」)
"""
import sys, os
sys.path.insert(0, os.environ.get("MSBENCH_PKGS", os.path.expanduser("~/metrics_pkgs")))   # lpips 另外裝的位置
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
_lp = _dino = _proc = None


def load(p, short=None):
    im = Image.open(p).convert("RGB")
    if short:
        k = short / min(im.size); im = im.resize((round(im.width * k), round(im.height * k)), Image.BICUBIC)
    return im


def t(im):  # [1,3,H,W] 0..1
    return torch.from_numpy(np.asarray(im)).permute(2, 0, 1)[None].float().div(255).to(DEV)


def ssim(a, b):
    g = lambda x: (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])[:, None]
    x, y = g(t(a)), g(t(b))
    k = torch.exp(-(torch.arange(11, device=DEV) - 5.0) ** 2 / (2 * 1.5 ** 2)); k = (k / k.sum())
    w = (k[:, None] * k[None, :])[None, None]
    mu = lambda z: F.conv2d(z, w)
    mx, my = mu(x), mu(y); sxx = mu(x * x) - mx ** 2; syy = mu(y * y) - my ** 2; sxy = mu(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))).mean().item()


def lap_var(im):
    x = torch.from_numpy(np.asarray(im.convert("L"))).float()[None, None].to(DEV) / 255
    return F.conv2d(x, torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float, device=DEV)).var().item()


def compare(ref_path, img_path):
    global _lp, _dino, _proc
    if _lp is None:
        import lpips
        _lp = lpips.LPIPS(net="alex", verbose=False).to(DEV)
        from transformers import AutoImageProcessor, AutoModel
        _proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        _dino = AutoModel.from_pretrained("facebook/dinov2-small").to(DEV).eval()
    R, I = load(ref_path), load(img_path)
    out = {"mae64": float(np.abs(np.asarray(R.resize((64, 64), Image.BILINEAR), float)
                                 - np.asarray(I.resize((64, 64), Image.BILINEAR), float)).mean())}
    a, b = load(ref_path, 512), load(img_path, 512)
    out["ssim"] = 1 - ssim(a, b)
    with torch.no_grad():
        out["lpips"] = _lp(t(a) * 2 - 1, t(b) * 2 - 1).item()
        f = _dino(**_proc(images=[R, I], return_tensors="pt").to(DEV)).last_hidden_state
        out["dino"] = 1 - F.cosine_similarity(f[0, 0], f[1, 0], dim=0).item()
        out["dinop"] = 1 - F.cosine_similarity(f[0, 1:], f[1, 1:], dim=-1).mean().item()
    out["sharp"] = lap_var(b) / lap_var(a)
    return out
