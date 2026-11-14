---
license: apache-2.0
tags:
  - lora
  - gguf
  - qwen3-coder
  - wcag
  - accessibility
  - tool-use
base_model: Qwen/Qwen3-Coder-30B-A3B-Instruct
---

# WCAG Gauntlet LoRA Adapter — Qwen3-Coder-30B-A3B (GGUF, F16)

LoRA adapter trained on the **WCAG Gauntlet tool-use skill** for
`Qwen/Qwen3-Coder-30B-A3B-Instruct`. Converts to an Ollama-ready GGUF
adapter (218 MB, F16, 672 non-expert LoRA tensor pairs).

> **Note:** This adapter covers **non-expert (attention + shared-MLP) layers only**.
> MoE expert LoRA pairs (the bulk of Qwen3-30B-A3B's 37k LoRA tensors) are not
> yet packed — llama.cpp upstream skips them for qwen3moe. A custom 3D expert
> packing pass is planned. For full expert coverage, use the raw PEFT adapter
> (`hummern/qwen3-coder-30b-a3b-wcag-lora-v1` on Kaggle) with `peft` directly.

## Files

| File | Size | Description |
|---|---|---|
| `wcag-gauntlet-30b-a3b-adapter.gguf` | 218 MB | GGUF LoRA adapter (F16) |

## How to run with Ollama

```bash
# 1. Pull the base model (quantized, ~18 GB for Q4_K_M)
ollama pull qwen3-coder:30b

# 2. Download this adapter
hf download Turbolego/wcag-gauntlet-30b-a3b-gguf-adapter   --local-dir ./wcag-adapter

# 3. Create a Modelfile
cat > Modelfile <<'EOF'
FROM qwen3-coder:30b
ADAPTER ./wcag-adapter/wcag-gauntlet-30b-a3b-adapter.gguf
PARAMETER num_ctx 262144
EOF

# 4. Build and run
ollama create wcag-gauntlet-30b -f Modelfile
ollama run wcag-gauntlet-30b "Audit this HTML for WCAG 2.2 AA violations"
```

## How to run with llama.cpp directly

```bash
# Base model (GGUF quant, e.g. Q4_K_M from bartowski or unsloth)
hf download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF   Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf --local-dir ./base

# Adapter (this repo)
hf download Turbolego/wcag-gauntlet-30b-a3b-gguf-adapter --local-dir ./adapter

llama-cli   -m ./base/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf   --lora ./adapter/wcag-gauntlet-30b-a3b-adapter.gguf   --ctx-size 262144   -p "List WCAG 2.2 Level AA contrast requirements"
```

## Compare with base model

Run the same prompt on both and diff:

```bash
# Base only
ollama run qwen3-coder:30b "Audit <div style='color:#ccc;background:#fff'>low contrast</div> for WCAG AA" > base.txt

# With adapter
ollama run wcag-gauntlet-30b "Audit <div style='color:#ccc;background:#fff'>low contrast</div> for WCAG AA" > adapter.txt

diff base.txt adapter.txt
```

With llama.cpp:

```bash
llama-cli -m base.gguf -p "PROMPT" > base.txt
llama-cli -m base.gguf --lora adapter.gguf -p "PROMPT" > adapter.txt
diff base.txt adapter.txt
```

### What to look for

- **Base model**: generic coding advice, may miss specific WCAG SC numbers (e.g. 1.4.3 contrast 4.5:1).
- **Adapter model**: should cite exact WCAG 2.2 success criteria, suggest concrete fixes (e.g. darken to `#767676`), and reference the Gauntlet tool-use patterns (structured checklists, severity labels).

## Training data

Trained on the WCAG Gauntlet tool-use skill dataset — structured WCAG 2.2
violation detection, remediation steps, and severity classification across
common web patterns (contrast, keyboard nav, ARIA, focus management, etc.).

## Limitations

- Expert (MoE) LoRA pairs not included in this GGUF — see note above.
- F16 precision (~2x size of Q8); use `--outtype q8_0` rebuild for smaller adapter.
- Context window set to 262144 via Modelfile `PARAMETER num_ctx`.
