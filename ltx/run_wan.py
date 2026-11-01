"""Wan2.2 TI2V-5B (+Turbo 4-step) image->video на CPU. Вход: item.json, выход: <od>/<name>.mp4"""
import json, os, sys, time, gc, resource
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from diffusers import WanImageToVideoPipeline, WanTransformer3DModel, AutoencoderKLWan
from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers
from diffusers.utils import export_to_video, load_image
from diffusers.video_processor import VideoProcessor

it = json.load(open(sys.argv[1])); od = sys.argv[2]; os.makedirs(od, exist_ok=True)
torch.set_num_threads(os.cpu_count()); torch.set_grad_enabled(False); T0 = time.time()


def log(*a):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.time() - T0:7.1f}s maxrss={rss:5.1f}GB]", *a, flush=True)


BASE = it.get("base", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
TURBO = it.get("turbo", "quanhaol/Wan2.2-TI2V-5B-Turbo")
W, H, N = it.get("w", 480), it.get("h", 832), it.get("frames", 49)
COMPUTE = it.get("compute", "fp32")  # fp32: веса в bf16, считаем в fp32 (на AMD без bf16 так быстрее)
log("size", W, H, N, "compute", COMPUTE)

# 1) текст -> эмбеддинги, энкодер сразу выгружаем
pipe = WanImageToVideoPipeline.from_pretrained(BASE, transformer=None, vae=None, torch_dtype=torch.bfloat16)
log("text encoder loaded")
pe, _ = pipe.encode_prompt(prompt=it["prompt"], do_classifier_free_guidance=False, max_sequence_length=512, device=torch.device("cpu"))
pe = pe.float(); del pipe; gc.collect()
log("prompt encoded", tuple(pe.shape))

# 2) VAE и латент первого кадра
vae = AutoencoderKLWan.from_pretrained(BASE, subfolder="vae", torch_dtype=torch.float32)
vae.enable_tiling() if hasattr(vae, "enable_tiling") and it.get("tiling", True) else None
sf_, tf_ = 16, 4
vp = VideoProcessor(vae_scale_factor=sf_)
img = load_image(it["image"]).convert("RGB").resize((W, H))
x_img = vp.preprocess(img, height=H, width=W).to(torch.float32).unsqueeze(2)  # [1,3,1,H,W]
z = vae.encode(x_img).latent_dist.mode()
mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1); std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
cond = (z - mean) / std
F_, h_, w_ = (N - 1) // tf_ + 1, H // sf_, W // sf_
log("image encoded", tuple(cond.shape), "latent", F_, h_, w_)

# 3) трансформер: базовый конфиг + веса Turbo (или базовые)
cfg = WanTransformer3DModel.load_config(BASE, subfolder="transformer")
if it.get("use_turbo", True):
    files = list_repo_files(TURBO); log("turbo files", files)
    fn = it.get("turbo_file") or next(f for f in files if f.endswith((".pt", ".pth", ".safetensors")) and "model" in f)
    p = hf_hub_download(TURBO, fn); log("turbo downloaded", fn, round(os.path.getsize(p) / 1e9, 2), "GB")
    if p.endswith(".safetensors"):
        from safetensors.torch import load_file; sd = load_file(p)
    else:
        sd = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
    if isinstance(sd, dict) and "generator" in sd and isinstance(sd["generator"], dict): sd = sd["generator"]
    if isinstance(sd, dict) and "generator_ema" in sd and isinstance(sd["generator_ema"], dict): sd = sd["generator_ema"]
    sd2 = {}
    for k, v in sd.items():
        k = k.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")
        if k.startswith("model."): k = k[6:]
        sd2[k] = v.to(torch.bfloat16)
    del sd; gc.collect()
    log("turbo keys", len(sd2), list(sd2)[:3])
    sd2 = convert_wan_transformer_to_diffusers(sd2)
    from accelerate import init_empty_weights
    with init_empty_weights():
        tr = WanTransformer3DModel.from_config(cfg)
    miss, unexp = tr.load_state_dict(sd2, strict=False, assign=True)
    log("loaded turbo; missing", len(miss), miss[:5], "unexpected", len(unexp), unexp[:5])
    del sd2; gc.collect()
    tr = tr.to(torch.bfloat16)
else:
    tr = WanTransformer3DModel.from_pretrained(BASE, subfolder="transformer", torch_dtype=torch.bfloat16)
tr.eval()
log("transformer ready")

if COMPUTE == "fp32":
    def pre(m, args, kwargs):
        for p_ in m.parameters(recurse=False): p_.data = p_.data.float()
        for b_ in m.buffers(recurse=False):
            if b_.is_floating_point(): b_.data = b_.data.float()
        f = lambda a: a.float() if torch.is_tensor(a) and a.is_floating_point() else a
        return tuple(f(a) for a in args), {k: f(v) for k, v in kwargs.items()}

    def post(m, args, out):
        for p_ in m.parameters(recurse=False): p_.data = p_.data.to(torch.bfloat16)
    for m in tr.modules():
        if any(True for _ in m.parameters(recurse=False)):
            m.register_forward_pre_hook(pre, with_kwargs=True); m.register_forward_hook(post)
    dt = torch.float32
else:
    dt = torch.bfloat16

# 4) 4 шага Turbo: x0-предсказание + повторное зашумление (как в оригинальном скрипте)
shift = it.get("shift", 8.0)
idx = it.get("steps_idx", [1000, 750, 500, 250])
sig = [(shift * (s / 1000)) / (1 + (shift - 1) * (s / 1000)) for s in idx]  # sigma(1-i/1000) со сдвигом
if it.get("sigmas"): sig = it["sigmas"]
log("sigmas", [round(s, 4) for s in sig])
g = torch.Generator().manual_seed(it.get("seed", 0))
x = torch.randn(1, cond.shape[1], F_, h_, w_, generator=g)
m = torch.ones(1, 1, F_, h_, w_); m[:, :, 0] = 0
x = (1 - m) * cond + m * x
for i, s in enumerate(sig):
    t0 = time.time(); t = s * 1000
    ts = (m[0, 0][:, ::2, ::2] * t).flatten()[None]
    v = tr(hidden_states=x.to(dt), timestep=ts, encoder_hidden_states=pe.to(dt), return_dict=False)[0].float()
    x0 = x - s * v
    x0 = (1 - m) * cond + m * x0
    if i < len(sig) - 1:
        s2 = sig[i + 1]; x = (1 - s2) * x0 + s2 * torch.randn(x0.shape, generator=g)
        x = (1 - m) * cond + m * x
    log(f"step {i + 1}/{len(sig)} sigma={s:.3f} took {time.time() - t0:.1f}s")
del tr; gc.collect()

# 5) декод
lat = x0 * std + mean
vid = vae.decode(lat, return_dict=False)[0]
frames = vp.postprocess_video(vid, output_type="pil")[0]
log("decoded", len(frames))
export_to_video(frames, f"{od}/{it['name']}.mp4", fps=it.get("fps", 24))
log("DONE")
