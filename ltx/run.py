"""LTX-Video 2B distilled, image->video на CPU. python run.py <item.json> <outdir>"""
import json, os, sys, time, gc, torch
from diffusers import LTXConditionPipeline, LTXVideoTransformer3DModel, AutoencoderKLLTXVideo
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_image
it = json.load(open(sys.argv[1])); od = sys.argv[2]; os.makedirs(od, exist_ok=True)
torch.set_num_threads(os.cpu_count()); t0 = time.time()
CK = it.get("ckpt", "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-2b-0.9.6-distilled-04-25.safetensors")
BASE = it.get("base", "Lightricks/LTX-Video-0.9.5")
dt = getattr(torch, it.get("dtype", "bfloat16"))
tr = LTXVideoTransformer3DModel.from_single_file(CK, torch_dtype=dt)
vae = AutoencoderKLLTXVideo.from_single_file(CK, torch_dtype=dt) if it.get("vae_from_ckpt", True) else None
kw = {"transformer": tr}
if vae is not None: kw["vae"] = vae
pipe = LTXConditionPipeline.from_pretrained(BASE, torch_dtype=dt, **kw)
pipe.vae.enable_tiling()
print("loaded", round(time.time() - t0), "s", flush=True)
img = load_image(it["image"])
W, H = it.get("w", 512), it.get("h", 768); n = it.get("frames", 49)
img = img.resize((W, H))
cond = LTXVideoCondition(image=img, frame_index=0)
args = dict(conditions=[cond], prompt=it["prompt"], negative_prompt=it.get("neg", "worst quality, inconsistent motion, blurry, jittery, distorted"),
            width=W, height=H, num_frames=n, num_inference_steps=it.get("steps", 8), guidance_scale=it.get("cfg", 1.0),
            decode_timestep=it.get("decode_timestep", 0.05), decode_noise_scale=it.get("decode_noise_scale", 0.025),
            image_cond_noise_scale=it.get("icns", 0.0), generator=torch.Generator().manual_seed(it.get("seed", 0)), output_type="pil")
if it.get("timesteps"): args["timesteps"] = it["timesteps"]
frames = pipe(**args).frames[0]
export_to_video(frames, f"{od}/{it['name']}.mp4", fps=it.get("fps", 24))
print("done", it["name"], round(time.time() - t0), "s", flush=True)
