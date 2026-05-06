# Parsing Executable Primitives for Spatial Reasoning

PEPS is an agentic, schema-constrained spatial reasoning framework. Given a spatial query and one or more images, PEPS first parses the query into a required set of executable spatial primitives, then acquires the corresponding values through geometry tools, computes the answer with deterministic code, and verifies the resulting trace.

![alt text](assets/overview.jpg)



### 1. Environment

```bash
conda create -n peps python=3.11
conda activate peps

pip install -e .
pip install -r requirements.txt
```

### 2. API Key

PEPS uses OpenAI-compatible LLM calls for Parser, Executor, Coder, and Verifier agents.

```bash
export OPENAI_API_KEY="your-openai-api-key"
export PEPS_OPENAI_MODEL="gpt-4o"
```

Optional OpenAI-compatible endpoint:

```bash
export OPENAI_BASE_URL="https://your-endpoint.com/v1"
export OPENAI_API_KEY="your-api-key"
export PEPS_OPENAI_MODEL="your-model-name"
```


### 1. VGGT for `reconstruct`
```bash
git clone https://github.com/facebookresearch/vggt.git vggt
pip install -e vggt
export PEPS_VGGT_REPO_DIR="$PWD/vggt"
```

Download weights:

```bash
mkdir -p checkpoints/vggt

# Requires: pip install huggingface_hub
huggingface-cli download facebook/VGGT-1B model.pt \
    --local-dir checkpoints/vggt \
    --local-dir-use-symlinks False

export PEPS_VGGT_MODEL_DIR="$PWD/checkpoints/vggt"
```

Direct download alternative:

```bash
curl -L \
  https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt \
  -o checkpoints/vggt/model.pt
```

### 2. GroundingDINO for local `detect`
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git GroundingDINO
pip install --no-build-isolation -e GroundingDINO
```
Download weights:

```bash
mkdir -p checkpoints/grounding_dino

# Requires: pip install huggingface_hub
huggingface-cli download ShilongLiu/GroundingDINO groundingdino_swint_ogc.pth \
    --local-dir checkpoints/grounding_dino \
    --local-dir-use-symlinks False

export PEPS_GROUNDING_DINO_MODEL_DIR="$PWD/third_party/GroundingDINO"
export PEPS_GROUNDING_DINO_CHECKPOINT="$PWD/checkpoints/grounding_dino/groundingdino_swint_ogc.pth"
```

### 3. Orient Anything for local `predict_obj_pose`

```bash
git clone https://github.com/SpatialVision/Orient-Anything.git Orient-Anything
pip install -r Orient-Anything/requirements.txt
```

Download weights:

```bash
mkdir -p checkpoints/orient_anything

# Requires: pip install huggingface_hub
huggingface-cli download Viglong/Orient-Anything \
    --local-dir checkpoints/orient_anything \
    --local-dir-use-symlinks False

export PEPS_ORIENTATION_ANYTHING_MODEL_DIR="$PWD/checkpoints/orient_anything"
```

Optional V2 checkpoint:

```bash
mkdir -p checkpoints/orient_anything_v2
huggingface-cli download Viglong/OriAnyV2_ckpt \
    --local-dir checkpoints/orient_anything_v2 \
    --local-dir-use-symlinks False
```

### 4. MoGe-2 for local `estimate_scale`

```bash
git clone https://github.com/microsoft/MoGe.git MoGe
pip install -e MoGe
```

Download weights:

```bash
mkdir -p checkpoints/moge/moge-2-vitl-normal

# Requires: pip install huggingface_hub
huggingface-cli download Ruicheng/moge-2-vitl-normal \
    --local-dir checkpoints/moge/moge-2-vitl-normal \
    --local-dir-use-symlinks False

export PEPS_MOGE_MODEL_DIR="$PWD/checkpoints/moge/moge-2-vitl-normal"
```

Smaller MoGe-2 checkpoints can be used by replacing the Hugging Face repo name, for example:

```bash
huggingface-cli download Ruicheng/moge-2-vitb-normal \
    --local-dir checkpoints/moge/moge-2-vitb-normal \
    --local-dir-use-symlinks False
```

### 5. EasyOCR for local `ocr`

```bash
pip install easyocr
```

Pre-download English OCR weights into the PEPS checkpoint directory:

```bash
mkdir -p checkpoints/easyocr

python -c "import easyocr; easyocr.Reader(['en'], model_storage_directory='checkpoints/easyocr')"

export PEPS_EASYOCR_MODEL_DIR="$PWD/checkpoints/easyocr"
```

Add more languages by changing the language list:

```bash
python -c "import easyocr; easyocr.Reader(['ch_sim','en'], model_storage_directory='checkpoints/easyocr')"
```


## Usage

All commands should be run from the project root.

### Run One Query

```bash
python -m peps.entrypoints.run_query \
    --query "Is the chair to the left of the table?" \
    --image /path/to/image.jpg \
    --model gpt-4o
```

### Batch Run

Input JSON:

```json
[
  {
    "id": "q0",
    "query": "Is the chair to the left of the table?",
    "images": ["/path/to/image.jpg"],
    "choices": ["yes", "no"]
  }
]
```

Command:

```bash
python -m peps.entrypoints.run_batch \
    --input /path/to/batch.json \
    --output peps/data/reports/batch_predictions.jsonl \
    --model gpt-4o \
    --continue-on-error
```

### Evaluation

```bash
python -m peps.entrypoints.run_eval \
    --input /path/to/eval_dataset.jsonl \
    --dataset-name my_dataset \
    --predictions-output peps/data/reports/predictions.jsonl \
    --report-output peps/data/reports/report.json \
    --model gpt-4o
```

### Inspect a Trace

```bash
python -m peps.entrypoints.inspect_trace \
    --trace peps/data/traces/<trace-file>.json \
    --show-code \
    --show-workspace
```




## Acknowledgements

We would like to thank the following works for their codebases:
* [GCA](https://github.com/gca-spatial-reasoning/gca)
* [MSSR](https://github.com/gyj155/mssr)
* [pySpatial](https://github.com/Zhanpeng1202/pySpatial)
* [VADAR](https://github.com/damianomarsili/VADAR)
* [VGGT](https://github.com/facebookresearch/vggt)
* [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
* [SAM2](https://github.com/facebookresearch/sam2)


