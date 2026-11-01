import json, os, sys, time, gc, resource
import torch
from diffusers import LTXConditionPipeline, LTXVideoTransformer3DModel, AutoencoderKLLTXVideo
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_image

it = json.load(open(sys.argv[1])); od = sys.argv[2]; os.makedirs(od, exist_ok=True)
torch.set_num_threads(os.cpu_count()); T0 = time.time()


def log(*a):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.time() - T0:7.1f}s maxrss={rss:5.1f}GB]", *a, flush=True)


CK = it.get("ckpt", "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-2b-0.9.6-distilled-04-25.safetensors")
BASE = it.get("base", "Lightricks/LTX-Video-0.9.5")
dt = getattr(torch, it.get("dtype", "float32"))
te_dt = getattr(torch, it.get("te_dtype", "bfloat16"))
W, H = it.get("w", 512), it.get("h", 768); n = it.get("frames", 49)
log("cpu", os.cpu_count(), "dtype", dt, "size", W, H, n)

# 1) текст-энкодер отдельно: считаем эмбеддинги и сразу освобождаем память (T5-XXL ~9.5 ГБ)
pipe = LTXConditionPipeline.from_pretrained(BASE, transformer=None, vae=None, torch_dtype=te_dt)
log("text encoder loaded")
with torch.no_grad():
    pe, pm, ne, nm = pipe.encode_prompt(prompt=it["prompt"], negative_prompt=it.get("neg", "worst quality, inconsistent motion, blurry, jittery, distorted"),
                                        do_classifier_free_guidance=it.get("cfg", 1.0) > 1.0, device="cpu", max_sequence_length=it.get("max_seq", 128))
pe = pe.to(dt); ne = ne.to(dt) if ne is not None else None
pipe.text_encoder = None; pipe.register_modules(text_encoder=None); gc.collect()
log("prompt encoded, text encoder freed")

# 2) трансформер и VAE из чекпойнта
tr = LTXVideoTransformer3DModel.from_single_file(CK, torch_dtype=dt)
vae = AutoencoderKLLTXVideo.from_single_file(CK, torch_dtype=dt)
pipe.register_modules(transformer=tr, vae=vae)
pipe.vae_spatial_compression_ratio = vae.spatial_compression_ratio
pipe.vae_temporal_compression_ratio = vae.temporal_compression_ratio
pipe.transformer_spatial_patch_size = tr.config.patch_size
pipe.transformer_temporal_patch_size = tr.config.patch_size_t
pipe.vae.enable_tiling()
log("transformer+vae loaded")

img = load_image(it["image"]).convert("RGB").resize((W, H))
cond = LTXVideoCondition(image=img, frame_index=0)
ts = [time.time()]


def cb(p, i, t, kw):
    ts.append(time.time()); log(f"step {i + 1} t={float(t):.3f} took {ts[-1] - ts[-2]:.1f}s"); return kw


args = dict(conditions=[cond], prompt_embeds=pe, prompt_attention_mask=pm, width=W, height=H, num_frames=n,
            num_inference_steps=it.get("steps", 8), guidance_scale=it.get("cfg", 1.0), decode_timestep=it.get("decode_timestep", 0.05),
            decode_noise_scale=it.get("decode_noise_scale", 0.025), image_cond_noise_scale=it.get("icns", 0.0),
            generator=torch.Generator().manual_seed(it.get("seed", 0)), output_type="pil", callback_on_step_end=cb)
if ne is not None: args.update(negative_prompt_embeds=ne, negative_prompt_attention_mask=nm)
if it.get("timesteps"): args["timesteps"] = it["timesteps"]
with torch.no_grad():
    frames = pipe(**args).frames[0]
log("decoded", len(frames), "frames")
export_to_video(frames, f"{od}/{it['name']}.mp4", fps=it.get("fps", 24))
log("DONE")
