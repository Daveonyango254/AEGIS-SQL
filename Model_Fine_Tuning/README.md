# AEGIS-SQL Model Fine-Tuning

Comprehensive fine-tuning framework for training multilingual code models on the BIRD benchmark for Text-to-SQL generation.

## Overview

This directory contains everything needed to fine-tune state-of-the-art code models for AEGIS-SQL:

- **3 Multilingual Models**: Phi-4 (14B), Qwen2.5-Coder (7B), DeepSeek-Coder (6.7B)
- **BIRD Benchmark**: High-quality filtered training set (6,601 samples)
- **LoRA Training**: Parameter-efficient fine-tuning (trains only ~0.1% of parameters)
- **Best Practices**: Early stopping, gradient accumulation, learning rate scheduling
- **Auto-Upload**: Pushes trained adapters to HuggingFace Hub

## Directory Structure

```
Model_Fine_Tuning/
├── config/                     # Model-specific configurations
│   ├── phi4_config.yaml
│   ├── qwen_config.yaml
│   └── deepseek_config.yaml
├── data/                       # Training data (auto-downloaded)
├── scripts/                    # Fine-tuning scripts
│   ├── download_data.py        # Download BIRD from HuggingFace
│   ├── prepare_data.py         # Format and split data
│   ├── finetune.py             # Generic fine-tuning script
│   ├── finetune_phi4.py        # Phi-4 wrapper
│   ├── finetune_qwen.py        # Qwen wrapper
│   ├── finetune_deepseek.py    # DeepSeek wrapper
│   ├── evaluate_model.py       # Evaluate on BIRD dev
│   └── upload_to_hub.py        # Upload to HuggingFace
├── output/                     # Evaluation results
├── checkpoints/                # Model checkpoints
│   ├── phi4/
│   ├── qwen/
│   └── deepseek/
├── requirements.txt            # Dependencies
├── run_full_pipeline.py        # Main orchestrator
└── README.md                   # This file
```

## Quick Start

### 1. Setup RunPod GPU Instance

**Why RunPod?** Fine-tuning requires a powerful GPU (A100 40GB recommended). RunPod provides affordable on-demand GPU rentals (~$0.80/hr for A100).

**Setup Steps:**

1. **Create RunPod Account**: https://runpod.io
2. **Launch Pod**:
   - Template: PyTorch 2.0+ with CUDA 11.8+
   - GPU: A100 (40GB) or A6000 (48GB)
   - Storage: 100GB+ (for models and data)
3. **Get SSH Access**: Copy SSH command from RunPod dashboard
4. **SSH into Pod** from your Windows machine:
   ```bash
   ssh -p <PORT> -i C:\Users\david\.ssh\id_ed25519 root@<POD_IP>
   ```

**Install Dependencies (in RunPod):**

```bash
cd /workspace
git clone https://github.com/Daveonyango254/AEGIS-SQL.git
cd AEGIS-SQL/Model_Fine_Tuning
pip install -r requirements.txt
```

### 2. Set Up Environment (in RunPod)

Add your HuggingFace token to the `.env` file:

```bash
# In RunPod terminal
cd /workspace/AEGIS-SQL

# Create/edit .env file
nano .env

# Add this line (replace with your actual token):
HF_HUB_TOKEN=your_token_here

# Save: Ctrl+O, Enter, Ctrl+X
```

Get your token from: https://huggingface.co/settings/tokens

### 3. Run Full Pipeline (On RunPod GPU)

**Fine-tune a single model:**

```bash
# Phi-4 (14B, best for complex reasoning) - ~6-8 hours on A100
cd /workspace/AEGIS-SQL/Model_Fine_Tuning
python run_full_pipeline.py --model phi4

# Qwen2.5-Coder (7B, balanced performance) - ~4-5 hours
python run_full_pipeline.py --model qwen

# DeepSeek-Coder (6.7B, fastest training) - ~3-4 hours
python run_full_pipeline.py --model deepseek
```

**Fine-tune all models:**

```bash
python run_full_pipeline.py --model all
```

**Just prepare data (no training):**

```bash
python run_full_pipeline.py --data_only
```

### 4. Download Results to Local Windows Machine

**After training completes, download checkpoints and evaluation results:**

```powershell
# On local Windows machine (PowerShell or CMD)
# Replace PORT and POD_IP with your RunPod details

# Download checkpoints
scp -r -P <PORT> -i C:\Users\david\.ssh\id_ed25519 root@<POD_IP>:/workspace/AEGIS-SQL/Model_Fine_Tuning/checkpoints "C:\Users\david\Downloads\model_checkpoints"

# Download evaluation results
scp -r -P <PORT> -i C:\Users\david\.ssh\id_ed25519 root@<POD_IP>:/workspace/AEGIS-SQL/Model_Fine_Tuning/output "C:\Users\david\Downloads\finetuning_results"
```

**Example with actual RunPod values:**
```powershell
# Download Phi-4 checkpoint
scp -r -P 13809 -i C:\Users\david\.ssh\id_ed25519 root@213.173.109.97:/workspace/AEGIS-SQL/Model_Fine_Tuning/checkpoints/phi4 "C:\Users\david\Downloads\phi4_checkpoint"

# Download evaluation results
scp -r -P 13809 -i C:\Users\david\.ssh\id_ed25519 root@213.173.109.97:/workspace/AEGIS-SQL/Model_Fine_Tuning/output "C:\Users\david\Downloads\finetuning_output"
```

## Step-by-Step Manual Execution (On RunPod GPU)

If you prefer to run each step manually on RunPod:

### Step 0: SSH into RunPod

```bash
# From local Windows machine
ssh -p <PORT> -i C:\Users\david\.ssh\id_ed25519 root@<POD_IP>

# Clone repo (first time only)
cd /workspace
git clone https://github.com/Daveonyango254/AEGIS-SQL.git
cd AEGIS-SQL
```

### Step 1: Download Training Data

```bash
cd /workspace/AEGIS-SQL/Model_Fine_Tuning/scripts
python download_data.py --output_dir ../data
```

This downloads BIRD23-train-filtered (6,601 high-quality samples) from HuggingFace.

### Step 2: Prepare Data

```bash
python prepare_data.py \
  --input ../data/bird_train_raw.json \
  --output_dir ../data \
  --eval_split 0.1
```

This formats the data into instruction-tuning format and splits by database:
- **Train**: 90% of databases (~5,941 samples)
- **Eval**: 10% of databases (~660 samples)

### Step 3: Fine-Tune Model (On RunPod GPU)

**Option A: Use wrapper scripts (recommended)**

```bash
cd /workspace/AEGIS-SQL/Model_Fine_Tuning/scripts
python finetune_phi4.py      # For Phi-4 (6-8 hours)
python finetune_qwen.py      # For Qwen (4-5 hours)
python finetune_deepseek.py  # For DeepSeek (3-4 hours)
```

**Option B: Use generic script with config**

```bash
python finetune.py --config ../config/phi4_config.yaml
```

**Monitor training:**
```bash
# Watch training logs
tail -f /workspace/AEGIS-SQL/Model_Fine_Tuning/checkpoints/phi4/runs/*/events.out.tfevents.*

# Or use TensorBoard
tensorboard --logdir=/workspace/AEGIS-SQL/Model_Fine_Tuning/checkpoints/phi4 --host=0.0.0.0 --port=6006
```

### Step 4: Evaluate on BIRD Dev Set

**Method 1: Using AEGIS evaluation pipeline (recommended)**

```bash
# Update config.yaml to use fine-tuned adapter
cd /workspace/AEGIS-SQL

# Edit config.yaml:
# slm:
#   model: "microsoft/Phi-4-mini-instruct"
#   adapter_path: "Model_Fine_Tuning/checkpoints/phi4"

# Run evaluation
python run_bird_evaluation.py \
  --bird_path data/bird \
  --num_queries 100 \
  --seed 42 \
  --output_name phi4_finetuned_eval
```

**Method 2: Using standalone evaluation script**

```bash
cd Model_Fine_Tuning/scripts
python evaluate_model.py \
  --adapter_path ../checkpoints/phi4 \
  --base_model microsoft/Phi-4-mini-instruct \
  --num_queries 100
```

Results saved to:
- Method 1: `evaluation/output/phi4_finetuned_eval/`
- Method 2: `Model_Fine_Tuning/output/<model>/`

**Download results to Windows:**

```powershell
# Replace PORT and POD_IP with your RunPod details
scp -r -P <PORT> -i C:\Users\david\.ssh\id_ed25519 root@<POD_IP>:/workspace/AEGIS-SQL/evaluation/output "C:\Users\david\Downloads\eval_results"

# Example:
scp -r -P 13809 -i C:\Users\david\.ssh\id_ed25519 root@213.173.109.97:/workspace/AEGIS-SQL/evaluation/output "C:\Users\david\Downloads\eval_results"
```

### Step 5: Upload to HuggingFace Hub

```bash
python upload_to_hub.py \
  --adapter_path ../checkpoints/phi4 \
  --repo_id Daveonyango254/aegis-sql-phi4-lora \
  --base_model microsoft/Phi-4-mini-instruct
```

## Model Configurations

### Phi-4 (14B)

- **Base**: microsoft/Phi-4-mini-instruct
- **Best for**: Complex reasoning, multilingual queries
- **Training**: ~6-8 hours on A100 (40GB)
- **Memory**: ~28GB VRAM with gradient checkpointing

### Qwen2.5-Coder (7B)

- **Base**: Qwen/Qwen2.5-Coder-7B-Instruct
- **Best for**: Balanced performance and speed
- **Training**: ~4-5 hours on A100
- **Memory**: ~18GB VRAM

### DeepSeek-Coder (6.7B)

- **Base**: deepseek-ai/deepseek-coder-6.7b-instruct
- **Best for**: Fast training, code-optimized
- **Training**: ~3-4 hours on A100
- **Memory**: ~16GB VRAM

## Training Hyperparameters

All models use the same best-practice hyperparameters:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| LoRA Rank | 16 | Balance between quality and efficiency |
| LoRA Alpha | 32 | Scaling factor (2*rank) |
| LoRA Dropout | 0.05 | Prevent overfitting |
| Epochs | 3 | Prevent overtraining |
| Learning Rate | 2e-4 | Conservative for stability |
| Warmup Ratio | 0.1 | 10% linear warmup |
| LR Scheduler | Cosine | Smooth decay |
| Batch Size (effective) | 16 | Via gradient accumulation |
| Weight Decay | 0.01 | L2 regularization |
| Early Stopping | Yes (patience=3) | Stop before overfitting |
| Mixed Precision | fp16/bf16 | 2x faster training |

## Best Practices Implemented

### 1. **Preventing Catastrophic Forgetting**

- ✅ **LoRA instead of full fine-tuning**: Only trains 0.1% of parameters
- ✅ **Conservative learning rate**: 2e-4 (vs 5e-5 for full fine-tune)
- ✅ **Early stopping**: Stops before base knowledge is degraded

### 2. **Ensuring Generalization**

- ✅ **Database-based split**: Eval uses different databases than train
- ✅ **Diverse training data**: 95 databases across 37 professional domains
- ✅ **Evidence-based prompting**: Teaches reasoning, not memorization
- ✅ **Schema variability**: Different table structures and column names

### 3. **Preventing Overfitting**

- ✅ **LoRA dropout** (0.05): Regularization in adapters
- ✅ **Weight decay** (0.01): L2 regularization
- ✅ **Early stopping** (patience=3): Monitors eval loss
- ✅ **Limited epochs** (3): Stops before overtraining

### 4. **Training Stability**

- ✅ **Gradient accumulation**: Stable updates with small batches
- ✅ **Learning rate warmup**: Prevents instability at start
- ✅ **Cosine decay**: Smooth convergence
- ✅ **Gradient clipping**: Max norm = 1.0

### 5. **Memory Efficiency**

- ✅ **Gradient checkpointing**: Trades compute for memory
- ✅ **Mixed precision training**: fp16/bf16 reduces memory
- ✅ **LoRA adaptation**: No need to store full model gradients

## Using Fine-Tuned Models

### In AEGIS-SQL

Update `config.yaml`:

```yaml
slm:
  model: "microsoft/Phi-4-mini-instruct"  # Base model
  adapter_path: "Daveonyango254/aegis-sql-phi4-lora"  # Your adapter
  device: "auto"
```

Then run AEGIS-SQL normally - it will automatically load the adapter.

### Standalone

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-4-mini-instruct",
    torch_dtype="auto",
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "Daveonyango254/aegis-sql-phi4-lora"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")

# Generate SQL
prompt = """Generate a valid SQLite query to answer the following question.

Schema:
CREATE TABLE employees (
  id INTEGER,
  name TEXT,
  salary REAL,
  department TEXT
);

Question: What is the average salary by department?

SQL Query:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128, temperature=0)
sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(sql)
```

## Troubleshooting

### Out of Memory (OOM)

1. **Reduce batch size** in config YAML:
   ```yaml
   training:
     per_device_train_batch_size: 1  # Reduce from 2 or 4
   ```

2. **Enable gradient checkpointing** (should already be on):
   ```yaml
   training:
     gradient_checkpointing: true
   ```

3. **Use smaller model**: Try DeepSeek-Coder (6.7B) instead of Phi-4 (14B)

### Training Too Slow

1. **Use mixed precision**:
   - Phi-4/Qwen: `bf16: true`
   - DeepSeek: `fp16: true`

2. **Increase batch size** (if you have memory):
   ```yaml
   training:
     per_device_train_batch_size: 8
     gradient_accumulation_steps: 2
   ```

3. **Use Flash Attention 2** (if supported):
   ```bash
   pip install flash-attn --no-build-isolation
   ```

### HuggingFace Upload Fails

1. **Check token**: Make sure `HF_HUB_TOKEN` is set in `.env`

2. **Verify permissions**: Token must have write access

3. **Check repo name**: Must be `username/repo-name` format

4. **Manual upload**:
   ```bash
   huggingface-cli login
   huggingface-cli upload Daveonyango254/aegis-sql-phi4-lora ./checkpoints/phi4
   ```

### Training Diverges (Loss → NaN)

1. **Reduce learning rate**:
   ```yaml
   training:
     learning_rate: 1.0e-4  # Reduce from 2e-4
   ```

2. **Check data quality**: Make sure `prepare_data.py` completed successfully

3. **Increase warmup**:
   ```yaml
   training:
     warmup_ratio: 0.2  # Increase from 0.1
   ```

## Expected Results

Based on BIRD benchmark performance:

| Model | Base EX | After Fine-Tuning (Expected) | Improvement |
|-------|---------|------------------------------|-------------|
| Phi-4 14B | ~35% | ~55-60% | +20-25pp |
| Qwen2.5-Coder 7B | ~32% | ~50-55% | +18-23pp |
| DeepSeek-Coder 6.7B | ~30% | ~48-52% | +18-22pp |

*Note: Results vary based on hardware, training time, and data quality.*

## RunPod Quick Reference

**Common Commands:**

```bash
# SSH into RunPod (from Windows)
ssh -p <PORT> -i C:\Users\david\.ssh\id_ed25519 root@<POD_IP>

# Check GPU status
nvidia-smi

# Monitor training
tail -f /workspace/AEGIS-SQL/Model_Fine_Tuning/checkpoints/phi4/training.log

# Download checkpoints (from Windows)
scp -r -P <PORT> -i C:\Users\david\.ssh\id_ed25519 root@<POD_IP>:/workspace/AEGIS-SQL/Model_Fine_Tuning/checkpoints/phi4 "C:\Users\david\Downloads\phi4_checkpoint"

# Download evaluation results (from Windows)
scp -r -P <PORT> -i C:\Users\david\.ssh\id_ed25519 root@<POD_IP>:/workspace/AEGIS-SQL/evaluation/output "C:\Users\david\Downloads\eval_results"

# Kill training (if needed)
pkill -f finetune

# Resume from checkpoint (if training crashed)
cd /workspace/AEGIS-SQL/Model_Fine_Tuning/scripts
python finetune_phi4.py --resume_from_checkpoint ../checkpoints/phi4/checkpoint-<STEP>
```

**Estimated Costs (RunPod A100 40GB @ $0.79/hr):**
- Phi-4 (6-8 hrs): ~$5-6
- Qwen (4-5 hrs): ~$3-4
- DeepSeek (3-4 hrs): ~$2-3
- All 3 models: ~$10-13

## Advanced: Custom Configuration

To create a custom configuration:

1. Copy an existing config (in RunPod):
   ```bash
   cd /workspace/AEGIS-SQL/Model_Fine_Tuning
   cp config/phi4_config.yaml config/my_custom_config.yaml
   ```

2. Modify parameters:
   ```yaml
   model:
     name: "your-model/your-model-name"

   lora:
     r: 32  # Increase rank for better quality (slower)

   training:
     num_epochs: 5  # More epochs (watch for overfitting!)
     learning_rate: 1.0e-4  # Lower LR for stability
   ```

3. Run with custom config:
   ```bash
   python scripts/finetune.py --config config/my_custom_config.yaml
   ```

## Citation

If you use these fine-tuned models in your research, please cite:

```bibtex
@software{aegis_sql_2025,
  title = {AEGIS-SQL: Privacy-Preserving Hybrid NL2SQL with Fine-Tuned Multilingual Models},
  author = {AEGIS Team},
  year = {2025},
  url = {https://github.com/yourusername/aegis_SQL}
}

@article{bird_benchmark,
  title = {BIRD: A Comprehensive Benchmark for Text-to-SQL},
  author = {Li, Jinyang and others},
  journal = {arXiv preprint arXiv:2305.03111},
  year = {2023}
}
```

## License

MIT License - See root LICENSE file for details.

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review logs in `Model_Fine_Tuning/checkpoints/<model>/`
3. Open an issue on GitHub with:
   - Model name and config
   - Error message
   - System specs (GPU, VRAM, RAM)

---

**Happy Fine-Tuning! 🚀**
