# Generative-AI-Engineering-and-Fine-Tuning-Transformers

Fine-tuning in machine learning is the process of adapting a pretrained model for specific tasks or use cases. During fine-tuning, the collate function tokenizes the dataset, the transformer-based model class defines classification in PyTorch, the forward method applies embeddings to the input, and the train_model function trains a transformer model. 

Fine-tuning enhances efficiency and saves time and computational resources compared to training models from scratch. It helps to transfer learning, time and resource efficiency, tailored responses, and task-specific adaptation.

HuggingFace is an open-source machine learning or ML platform with a built-in transformers library for natural language processing (or NLP) applications. Its built-in datasets can be loaded using the load_dataset function.

**Transformers and Fine Tuning :**
- Loading Models and Inference with Hugging Face Inferences
- [Optional] Pre-training LLMs with Hugging Face
- Pre-Training and Fine-Tuning with PyTorch
- Fine-Tuning Transformers with PyTorch and Hugging Face

**Parameter Efficient Fine Tuning (PEFT) :**
- Adapters with PyTorch
- LoRA with PyTorch
- [Optional] Lab: QLoRA with Hugging Face


Benefits of fine-tuning:

- Enhances efficiency and saves time 
- Transfers learning, time, and resource efficiency
- Tailors responses and task-specific adaptation
- Addresses issues like overfitting, underfitting, catastrophic forgetting, and data leakage

Approaches of fine-tuning language models:

- Self-supervised fine-tuning
- Supervised fine-tuning
- Reinforcement learning from human feedback
- Direct preference optimization

Hugging Face’s built-in data sets can be loaded using the load_dataset function. The tokenizer function extracts the text from the data set example and applies the tokenizer. The evaluation function evaluates the model’s performance after fine-tuning it.

SFT Trainer (or supervised fine-tuning trainer) simplifies and automates many training tasks, making the process more efficient and less error-prone compared to training with PyTorch directly.

Parameter-efficient fine-tuning (PEFT) methods reduce the number of trainable parameters that should be updated to adapt a large pretrained model to specific downstream applications effectively. 

Methods of PEFT are selective, additive, and reparameterization fine-tuning.

Soft prompts are learnable tensors concatenated with the input embedding that can be optimized to a data set; however, ranks minimize the number of vectors for space spanning.

LoRA helps complex ML for specific uses by adding lightweight plug-in components to the original model. It reduces the number of trainable parameters using pretrained models and matrix algebra to decompose weight updates into low-rank matrices.

In LoRA with PyTorch, the model uses the internet movie database (IMDB) data set and the class to create iterators for training and testing data sets; however, using the IMDB dataset and LoRA with Hugging Face simplifies the model training process.

QLoRA is a fine-tuning technique in ML for optimizing performance; however, quantization reduces the precision of numerical values to a finite set of discrete levels by defining the quantization range and levels. 

Model quantization reduces the precision of model parameters by reducing the model size and improving inference speed by maintaining the model’s accuracy. 

Some of the model quantization techniques are:

Uniform quantization

Non-uniform quantization

Weight clustering

Pruning

