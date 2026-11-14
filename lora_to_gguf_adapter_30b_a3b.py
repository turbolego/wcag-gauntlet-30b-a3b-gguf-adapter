"""Convert the trained Qwen3-30B-A3B PEFT LoRA adapter into an Ollama-ready GGUF adapter.

CPU-only (no GPU needed). Steps:
 1. read the PEFT adapter (safetensors + config) from the private dataset mount
 2. download base config/tokenizer (NO 62GB of weights)
 3. clone llama.cpp, run convert_lora_gguf.py (F16)
 4. VERIFY coverage: count .lora_a/.lora_b tensors in the result vs the PEFT
    adapter. llama.cpp upstream skips MoE expert LoRA pairs for qwen3moe
    (known gap) — if experts are missing, exit with a clear marker so the
    next kernel version can do custom 3D expert packing.

Output: /kaggle/working/wcag-gauntlet-30b-a3b-adapter.gguf  (~1.7 GB)
Ollama Modelfile:
    FROM qwen3-coder:30b
    ADAPTER ./wcag-gauntlet-30b-a3b-adapter.gguf
    PARAMETER num_ctx 262144
"""
import os, shutil, subprocess, sys, json

import glob
ADAPTER_DIR = "/kaggle/input/datasets/hummern/qwen3-coder-30b-a3b-wcag-lora-v1"
if not os.path.isdir(ADAPTER_DIR):
    g = glob.glob("/kaggle/input/datasets/*/*") + glob.glob("/kaggle/input/*/*")
    print("glob:", g, flush=True)
    ADAPTER_DIR = g[0] if g else ADAPTER_DIR
    if not os.path.isdir(ADAPTER_DIR):
        raise SystemExit("adapter dataset mount missing")
HF_REPO = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
HF_DIR = "/tmp/hf/qwen3-coder-30b"
OUT_GGUF = "/kaggle/working/wcag-gauntlet-30b-a3b-adapter.gguf"


def sh(*cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def main():
    global ADAPTER_DIR
    print("Adapter dir exists:", os.path.isdir(ADAPTER_DIR), flush=True)
    print("Contents:", os.listdir(ADAPTER_DIR) if os.path.isdir(ADAPTER_DIR) else "MISSING", flush=True)
    if not os.path.isdir(ADAPTER_DIR):
        raise SystemExit("adapter dataset mount missing — check dataset_sources in kernel metadata")

    # ALWAYS re-download fresh adapter from v23 kernel output (dataset copy may be corrupt)
    import struct, base64, urllib.request
    sf = os.path.join(ADAPTER_DIR, "adapter_model.safetensors")
    if os.path.exists(sf):
        sz = os.path.getsize(sf)
        with open(sf, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
        print(f"existing safetensors: size={sz} header={n}", flush=True)
    print("Re-downloading fresh adapter_model.safetensors via kaggle CLI...", flush=True)
    import subprocess as sp
    r = sp.run(["kaggle", "kernels", "output", "hummern/lora-tooluse-skill-30b-a3b-v23",
                "-p", "/tmp/v23out", "-o", "-q",
                "--file-pattern", "adapter_model.safetensors"], capture_output=True, text=True)
    print("kaggle CLI rc:", r.returncode, flush=True)
    print("stdout:", r.stdout[-500:], flush=True)
    print("stderr:", r.stderr[-500:], flush=True)
    # Find the downloaded file
    import glob as gl
    cands = gl.glob("/tmp/v23out/**/adapter_model.safetensors", recursive=True)
    print("candidates:", cands, flush=True)
    if not cands:
        raise SystemExit("download failed")
    fresh_dir = "/tmp/adapter_fresh"
    os.makedirs(fresh_dir, exist_ok=True)
    shutil.copy(cands[0], os.path.join(fresh_dir, "adapter_model.safetensors"))
    shutil.copy(os.path.join("/kaggle/input/datasets/hummern/qwen3-coder-30b-a3b-wcag-lora-v1", "adapter_config.json"),
                os.path.join(fresh_dir, "adapter_config.json"))
    print("fresh adapter in", fresh_dir, "size:", os.path.getsize(os.path.join(fresh_dir, "adapter_model.safetensors")), flush=True)
    ADAPTER_DIR = fresh_dir

    sh(sys.executable, "-m", "pip", "install", "-q", "gguf", "safetensors", "numpy", "huggingface_hub")

    os.makedirs(HF_DIR, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(
        HF_REPO,
        allow_patterns=["config.json", "tokenizer*.json", "tokenizer*.model", "tokenizer_config.json",
                        "special_tokens_map.json", "generation_config.json", "chat_template.jinja"],
        local_dir=HF_DIR,
    )
    print("Base config+tokenizer downloaded ->", os.listdir(HF_DIR), flush=True)

    if not os.path.isdir("/tmp/llama.cpp"):
        sh("git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", "/tmp/llama.cpp")
    import glob as _glob
    candidates = _glob.glob("/tmp/llama.cpp/**/convert_lora*.py", recursive=True)
    print("converter candidates:", candidates, flush=True)
    conv = next((c for c in candidates if "to_gguf" in c), candidates[0] if candidates else None)
    if not conv:
        raise SystemExit("no convert_lora*.py found in llama.cpp checkout")
    print("RUNNING:", sys.executable, conv, ADAPTER_DIR, "--base", HF_DIR, flush=True)
    r = subprocess.run([sys.executable, conv, ADAPTER_DIR,
                        "--base", HF_DIR,
                        "--outtype", "f16", "--outfile", OUT_GGUF], capture_output=True, text=True)
    print("RC:", r.returncode, flush=True)
    print("STDOUT:", r.stdout[-3000:], flush=True)
    print("STDERR:", r.stderr[-4000:], flush=True)
    if r.returncode != 0:
        raise SystemExit(f"converter failed rc={r.returncode}")

    # --- coverage verification ---
    import gguf
    from safetensors import safe_open
    reader = gguf.GGUFReader(OUT_GGUF)
    n_gguf = len(reader.tensors)
    print(f"GGUF adapter tensors: {n_gguf}", flush=True)

    sf = os.path.join(ADAPTER_DIR, "adapter_model.safetensors")
    with safe_open(sf, framework="pt") as f:
        peft_keys = [k for k in f.keys() if "lora_A" in k or "lora_B" in k]
    print(f"PEFT lora tensors: {len(peft_keys)}", flush=True)

    with open(os.path.join(ADAPTER_DIR, "adapter_coverage.json"), "w") as fh:
        json.dump({
            "gguf_tensors": n_gguf,
            "peft_lora_tensors": len(peft_keys),
            "sample_peft_keys": peft_keys[:15],
            "expert_keys_in_peft": [k for k in peft_keys if "experts" in k][:15],
            "gguf_expert_tensors": [t.name for t in reader.tensors if "expert" in t.name.lower() or "ffn" in t.name.lower()][:20],
        }, fh, indent=2)
    print(open(os.path.join(ADAPTER_DIR, "adapter_coverage.json")).read()[:2000], flush=True)

    if len(peft_keys) > 0 and n_gguf < len(peft_keys) * 0.5:
        raise SystemExit(f"EXPERT_GAP: converter produced {n_gguf} tensors for {len(peft_keys)} PEFT lora tensors — "
                         "expert pairs were skipped; custom 3D expert packing needed (next kernel version)")
    size_mb = os.path.getsize(OUT_GGUF) / 1024 / 1024
    print(f"OK: {OUT_GGUF} ({size_mb:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
