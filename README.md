# Generative-AI-Engineering-and-Fine-Tuning-Transformers



**Transformers and Fine Tuning :**
- Loading Models and Inference with Hugging Face Inferences
- [Optional] Pre-training LLMs with Hugging Face
- Pre-Training and Fine-Tuning with PyTorch
- Fine-Tuning Transformers with PyTorch and Hugging Face

**Parameter Efficient Fine Tuning (PEFT) :**
- Adapters with PyTorch
- LoRA with PyTorch
- [Optional] Lab: QLoRA with Hugging Face


**Fine-Tuning Transformers**
This repository covers transformer fine-tuning and parameter-efficient techniques for adapting large language models to specific tasks.


**Topics Covered**


**Transformers & Fine-Tuning:**
Loading models and inference with Hugging Face
Pre-training and fine-tuning with PyTorch
Building custom training loops

**Parameter-Efficient Fine-Tuning (PEFT):**

- Adapters - Small trainable layers inserted between frozen transformer blocks
- LoRA - Low-rank adaptation that decomposes weight updates into smaller matrices
- QLoRA - Quantized LoRA for even more memory-efficient training

**Key Concepts**
**Fine-tuning** adapts pretrained models for specific tasks by updating weights on task-specific data, saving time and resources compared to training from scratch.
**PEFT methods** reduce trainable parameters while maintaining performance:

- Update only 1-5% of model parameters
- Enable fine-tuning on consumer hardware
- Can be "plugged in/out" of base models

Approaches

- Self-supervised fine-tuning
- Supervised fine-tuning
- Reinforcement learning from human feedback (RLHF)
- Direct preference optimization (DPO)

Tools & Libraries
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT library
