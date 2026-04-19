# vLLM + Llama 3.1 on an RTX 3090

This guide shows how to run Llama 3.1 with vLLM on a separate RTX 3090 ML server and connect this RAG repo to it.

The repo talks to vLLM through its OpenAI-compatible `/v1/chat/completions` API.

## Architecture

```text
RAG repo machine
  - PDF ingestion
  - embeddings
  - ChromaDB retrieval
  - Streamlit or CLI
        |
        | HTTP
        v
RTX 3090 ML server
  - vLLM
  - Llama 3.1 Instruct
  - OpenAI-compatible API
```

## Recommended Model for a 3090

Use:

```text
meta-llama/Meta-Llama-3.1-8B-Instruct
```

An RTX 3090 has 24 GB VRAM. Llama 3.1 8B Instruct is the practical default. Llama 3.1 70B and 405B are not realistic on a single 3090 without heavy quantization, offloading, or multiple GPUs.

The Llama 3.1 Hugging Face model is gated, so accept the model terms on Hugging Face first and make sure the ML server has a valid Hugging Face token.

## 1. Prepare the 3090 ML Server

On the ML server:

```bash
nvidia-smi
```

Confirm the RTX 3090 is visible.

Create an isolated environment:

```bash
conda create -n vllm-llama python=3.12 -y
conda activate vllm-llama
pip install --upgrade pip
pip install vllm
```

Log in to Hugging Face:

```bash
pip install huggingface_hub
huggingface-cli login
```

Paste a Hugging Face token that has access to `meta-llama/Meta-Llama-3.1-8B-Instruct`.

## 2. Start vLLM on the 3090

Pick an API key for your local network. It can be any strong random string.

```bash
export VLLM_API_KEY="replace-with-a-long-random-token"
```

Start the server:

```bash
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --api-key "$VLLM_API_KEY"
```

Notes:

- `--host 0.0.0.0` lets another machine on your LAN connect.
- `--max-model-len 8192` is a conservative starting point for a 24 GB 3090.
- If you run out of VRAM, lower `--max-model-len` or `--gpu-memory-utilization`.
- If you want vLLM defaults instead of model repository generation defaults, add `--generation-config vllm`.

## 3. Test vLLM on the ML Server

In another terminal on the ML server:

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $VLLM_API_KEY"
```

Then test chat completion:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "Reply with one short sentence."}
    ],
    "temperature": 0.1
  }'
```

## 4. Connect from This Repo Over the LAN

Find the ML server IP address:

```bash
hostname -I
```

Assume it is:

```text
192.168.1.50
```

On the RAG repo machine, edit `.env`:

```text
LLM_BACKEND=vllm
LLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
LLM_BASE_URL=http://192.168.1.50:8000/v1
LLM_API_KEY=replace-with-a-long-random-token
LLM_TEMPERATURE=0.1
```

Then test from the repo machine:

```bash
curl http://192.168.1.50:8000/v1/models \
  -H "Authorization: Bearer replace-with-a-long-random-token"
```

Run the RAG query:

```bash
python scripts/query.py "What does the SOP say about maintenance?"
```

Or start the UI:

```bash
streamlit run app/streamlit_app.py
```

## 5. Connect with an SSH Tunnel

Use this if the 3090 server is reachable by SSH but you do not want to expose port `8000` on the LAN.

On the RAG repo machine:

```bash
ssh -L 8000:127.0.0.1:8000 user@ML_SERVER_HOSTNAME_OR_IP
```

Keep that SSH session open.

Set `.env` in this repo:

```text
LLM_BACKEND=vllm
LLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
LLM_BASE_URL=http://127.0.0.1:8000/v1
LLM_API_KEY=replace-with-a-long-random-token
LLM_TEMPERATURE=0.1
```

Now this repo connects to the 3090 through the tunnel:

```bash
python scripts/query.py "What warnings are listed?"
```

## 6. Full Repo Flow

From this repo:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add PDFs:

```text
docs/
```

Ingest:

```bash
python scripts/ingest.py
```

Query using vLLM on the 3090:

```bash
python scripts/query.py "How should cleaning be performed?"
```

Run the UI:

```bash
streamlit run app/streamlit_app.py
```

## 7. Troubleshooting

### `401 Unauthorized`

The `LLM_API_KEY` in this repo does not match the `--api-key` used when starting vLLM.

### `Connection refused`

Check that vLLM is running and that the URL includes `/v1`:

```text
LLM_BASE_URL=http://ML_SERVER_IP:8000/v1
```

### Server works locally but not from the repo machine

Check firewall rules on the ML server:

```bash
sudo ufw status
```

Allow the port if needed:

```bash
sudo ufw allow from RAG_REPO_MACHINE_IP to any port 8000 proto tcp
```

### CUDA out of memory

Try:

```bash
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --api-key "$VLLM_API_KEY"
```

### Model access denied

Accept the Llama 3.1 license terms on Hugging Face and log in again:

```bash
huggingface-cli login
```

## References

- vLLM OpenAI-compatible server docs: https://docs.vllm.ai/en/latest/serving/openai_compatible_server/
- vLLM installation docs: https://docs.vllm.ai/en/latest/getting_started/installation/
- Llama 3.1 8B Instruct model card: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
