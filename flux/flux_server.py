"""FLUX picture server for one RunPod session (Python stdlib HTTP + diffusers). Port 8000.
Every request needs the header X-AF-Key = $AF_POD_KEY (a random key made by the app for this pod only).
  GET  /health                 {"ready": {"schnell": bool, "kontext": bool}, "step": "...", "error": "..."}
  POST /schnell  {prompt, seed, width, height, steps=4}                 -> {"id"}
  POST /kontext  {image_b64, prompt, seed, guidance=2.5, steps=28}      -> {"id"}
  GET  /result/ID              {"status": "queued|running|done|failed", "png_b64"?, "secs"?, "error"?}
Requests are queued and run one at a time (async, so the RunPod proxy's 100 s limit never matters).
Env: AF_POD_KEY (required), HF_TOKEN (needed for the gated Kontext repo), AF_NEED="schnell,kontext" (what to load at start),
AF_FAKE=1 (no GPU: returns grey pictures, for testing the app side)."""
import base64, io, json, os, queue, threading, time, traceback, uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

KEY = os.environ.get("AF_POD_KEY") or ""
FAKE = os.environ.get("AF_FAKE") == "1"
BASE = os.environ.get("AF_SCHNELL_REPO", "unsloth/FLUX.1-schnell")            # Apache-2.0 copy of FLUX.1 schnell (not gated)
KONTEXT = os.environ.get("AF_KONTEXT_REPO", "black-forest-labs/FLUX.1-Kontext-dev")   # gated: the HF account must accept its licence
LOCK_FILE = "/workspace/.job_running"
S = {"ready": {"schnell": False, "kontext": False}, "step": "starting", "error": "", "t0": time.time(), "done": 0}
P = {}                    # loaded pipelines and shared parts
JOBS, Q = {}, queue.Queue()
ML = threading.Lock()     # model loading / GPU use


def step(m):
    S["step"] = m; print(time.strftime("%H:%M:%S"), m, flush=True)


def busy(on):
    try:
        if on: open(LOCK_FILE, "w").write("busy")
        elif os.path.exists(LOCK_FILE): os.remove(LOCK_FILE)
    except OSError: pass


def load(kind):
    """Load schnell or kontext. Text encoders + VAE are shared and stay on the GPU; only one 12B transformer sits on the GPU
    at a time (the other waits in RAM), so both fit a 48 GB card."""
    if FAKE:
        time.sleep(1); S["ready"][kind] = True; return
    import torch
    from huggingface_hub import snapshot_download
    from diffusers import FluxPipeline, FluxKontextPipeline, FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler
    tok = os.environ.get("HF_TOKEN") or None
    with ML:
        if S["ready"][kind]: return
        if "base" not in P:
            step("downloading the shared FLUX parts (text encoders, VAE, ~10 GB)")
            d = snapshot_download(BASE, allow_patterns=["model_index.json", "scheduler/*", "text_encoder/*", "text_encoder_2/*",
                                                       "tokenizer/*", "tokenizer_2/*", "vae/*", "transformer/config.json"], max_workers=16)
            step("loading the shared parts")
            base = FluxPipeline.from_pretrained(d, transformer=None, torch_dtype=torch.bfloat16)
            for m in (base.text_encoder, base.text_encoder_2, base.vae): m.to("cuda")
            P["base"], P["base_dir"] = base, d
        b = P["base"]
        if kind == "schnell":
            step("downloading FLUX.1 schnell (~24 GB)")
            d = snapshot_download(BASE, allow_patterns=["transformer/*"], max_workers=16)
            step("loading FLUX.1 schnell")
            tr = FluxTransformer2DModel.from_pretrained(d, subfolder="transformer", torch_dtype=torch.bfloat16)
            P["schnell"] = FluxPipeline(scheduler=b.scheduler, vae=b.vae, text_encoder=b.text_encoder, tokenizer=b.tokenizer,
                                        text_encoder_2=b.text_encoder_2, tokenizer_2=b.tokenizer_2, transformer=tr)
        else:
            step("downloading FLUX.1 Kontext [dev] (~24 GB)")
            d = snapshot_download(KONTEXT, allow_patterns=["model_index.json", "scheduler/*", "transformer/*"], token=tok, max_workers=16)
            step("loading FLUX.1 Kontext [dev]")
            tr = FluxTransformer2DModel.from_pretrained(d, subfolder="transformer", torch_dtype=torch.bfloat16)
            sch = FlowMatchEulerDiscreteScheduler.from_pretrained(d, subfolder="scheduler")
            P["kontext"] = FluxKontextPipeline(scheduler=sch, vae=b.vae, text_encoder=b.text_encoder, tokenizer=b.tokenizer,
                                               text_encoder_2=b.text_encoder_2, tokenizer_2=b.tokenizer_2, transformer=tr)
        P[kind].transformer.to("cpu"); S["ready"][kind] = True
        step(f"{kind} ready")


def on_gpu(kind):
    other = "kontext" if kind == "schnell" else "schnell"
    if other in P and next(P[other].transformer.parameters()).device.type == "cuda": P[other].transformer.to("cpu")
    if next(P[kind].transformer.parameters()).device.type != "cuda": P[kind].transformer.to("cuda")


def make(kind, a):
    if FAKE:
        from PIL import Image
        time.sleep(0.5); im = Image.new("RGB", (int(a.get("width", 768)), int(a.get("height", 1024))), (128, 128, 128))
    else:
        import torch
        if not S["ready"][kind]: load(kind)
        with ML:
            on_gpu(kind); g = torch.Generator("cuda").manual_seed(int(a.get("seed", 7)))
            if kind == "schnell":
                im = P["schnell"](prompt=a["prompt"], width=int(a.get("width", 1024)), height=int(a.get("height", 1024)),
                                  num_inference_steps=int(a.get("steps", 4)), guidance_scale=0.0, generator=g).images[0]
            else:
                from PIL import Image
                src = Image.open(io.BytesIO(base64.b64decode(a["image_b64"]))).convert("RGB")
                im = P["kontext"](image=src, prompt=a["prompt"], guidance_scale=float(a.get("guidance", 2.5)),
                                  num_inference_steps=int(a.get("steps", 28)), generator=g).images[0]
    buf = io.BytesIO(); im.save(buf, "PNG"); return base64.b64encode(buf.getvalue()).decode()


def worker():
    while True:
        jid = Q.get(); j = JOBS[jid]; j["status"] = "running"; t = time.time(); busy(True)
        try:
            j["png_b64"] = make(j.pop("kind"), j.pop("args")); j["status"] = "done"; S["done"] += 1
        except Exception as e:
            traceback.print_exc(); j["status"] = "failed"; j["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        j["secs"] = round(time.time() - t, 1); busy(not Q.empty())


def preload():
    busy(True)
    try:
        for k in [x.strip() for x in os.environ.get("AF_NEED", "schnell,kontext").split(",") if x.strip() in ("schnell", "kontext")]:
            load(k)
        step("ready")
    except Exception as e:
        traceback.print_exc(); S["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        if "gated" in str(e).lower() or "403" in str(e): S["error"] = "GATED: the Hugging Face account has not accepted the FLUX.1 Kontext [dev] licence"
        step("load failed")
    busy(False)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def ok(self):
        if KEY and self.headers.get("X-AF-Key") == KEY: return True
        self.send(403, {"error": "wrong key"}); return False

    def do_GET(self):
        if not self.ok(): return
        if self.path == "/health":
            return self.send(200, {"ready": S["ready"], "step": S["step"], "error": S["error"], "up_s": round(time.time() - S["t0"]),
                                   "queued": Q.qsize(), "done": S["done"]})
        if self.path.startswith("/result/"):
            j = JOBS.get(self.path.split("/")[-1])
            if not j: return self.send(404, {"error": "unknown id"})
            out = {k: v for k, v in j.items() if k in ("status", "png_b64", "secs", "error")}
            if j["status"] in ("done", "failed"): JOBS.pop(self.path.split("/")[-1], None)
            return self.send(200, out)
        self.send(404, {"error": "not found"})

    def do_POST(self):
        if not self.ok(): return
        kind = self.path.strip("/")
        if kind not in ("schnell", "kontext"): return self.send(404, {"error": "not found"})
        try: a = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except Exception: return self.send(400, {"error": "bad json"})
        if not a.get("prompt") or (kind == "kontext" and not a.get("image_b64")): return self.send(400, {"error": "prompt (and image for kontext) required"})
        jid = uuid.uuid4().hex[:12]; JOBS[jid] = {"status": "queued", "kind": kind, "args": a}; Q.put(jid)
        self.send(200, {"id": jid})


if __name__ == "__main__":
    if not KEY: raise SystemExit("AF_POD_KEY is not set: refusing to serve without a key")
    threading.Thread(target=preload, daemon=True).start()
    threading.Thread(target=worker, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8000"))), H).serve_forever()
