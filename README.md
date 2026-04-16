# 🌸 Generative Models & Multimodal Learning Project

## 📌 Problem Statement
This project explores modern generative AI techniques across images and text, focusing on:

- Understanding and analyzing a real-world dataset (**Oxford Flowers**)
- Fine-tuning a large-scale generative model (**Stable Diffusion 1.5**) using **LoRA**
- Building a text preprocessing pipeline using **CLIP**
- Designing a **Conditional GAN (CGAN)** for controlled image generation

The goal is to bridge:
- **Computer Vision (images)**
- **Natural Language Processing (text)**
- **Generative Modeling (diffusion + GANs)**
  
## 📂 Dataset

### 🌼 Oxford 102 Flowers Dataset
- **Total Images:** 1020 (train split)  
- **Classes:** 102 flower categories  
- **Average Resolution:** ~629 × 531 px  
- **Width Range:** 500 – 919 px  
- **Height Range:** 500 – 930 px  
- **Aspect Ratio (avg):** 1.21  

### 🎨 Pixel Statistics
- **Mean (RGB):**
  - R: 0.423  
  - G: 0.375  
  - B: 0.291  
- **Std (RGB):**
  - R: 0.121  
  - G: 0.112  
  - B: 0.129  

### 📝 Text Descriptions
- **Total descriptions:** 102  
- **Avg length:** ~12.5 words  
- Used to create text prompts for diffusion training  

## ⚙️ Methodology

The project is divided into **6 tasks**, each targeting a different concept:

### 🔹 Task 1 & 2 (Base Training + Generator)
- Built a base generative pipeline  
- Established training and evaluation workflow  

### 🔹 Task 3: Stable Diffusion Fine-tuning (LoRA)
**Model**
- Base: Stable Diffusion v1.5  
- Components:  
  - VAE (frozen)  
  - CLIP Text Encoder (frozen)  
  - UNet (LoRA fine-tuned)  

**LoRA Configuration**
- Rank (r): 4  
- Trainable Params: 398,592 (~0.05%)  
- Total Params: ~860M  

**Training Setup**
- Steps: 300  
- Learning Rate: 1e-4  
- Gradient Accumulation: 4  
- Effective Batch Size: 4  

**Pipeline**
1. Encode images → latent space (VAE)  
2. Add noise (DDPM scheduler)  
3. Condition on text embeddings (CLIP)  
4. Predict noise (UNet + LoRA)  
5. Optimize using MSE loss  

### 🔹 Task 4: Dataset Exploration & Analysis
Performed deep **EDA** on the dataset:

- **Analysis Includes:**
  - Image resolution distribution  
  - Aspect ratio histogram  
  - Class distribution  
  - RGB channel statistics  
  - Text description analysis (word count, frequent words)  

- **Visualization Outputs**

### 🔹 Task 5: Text Preprocessing (CLIP)
**Model**
- CLIP tokenizer + text encoder (from Stable Diffusion)  

**Steps**
- Tokenization  
- Attention masking  
- Embedding generation  

**Key Observations**
- Prompts mapped to fixed-length token sequences (77 tokens)  
- Output embeddings: 768-dimensional vectors  
- Semantic similarity captured using cosine similarity  

**Outputs**
- Embedding heatmaps  
- Similarity matrix  
- Token breakdown per prompt  

### 🔹 Task 6: Conditional GAN (CGAN)
**Dataset**
- Synthetic shapes: Circle, Square, Triangle (1500 images total)  

**Architecture**
- **Generator:** Input (Noise + Label embedding) → Output (32×32 image)  
- **Discriminator:** Input (Image + Label embedding) → Output (Real/Fake score)  

**Training**
- Epochs: 100  
- Loss: Binary Cross Entropy  
- Optimizer: Adam  

**Conditioning Mechanism**
- Labels embedded and concatenated with:  
  - Noise (Generator)  
  - Image features (Discriminator)  

## 📊 Results

### 🔹 Task 3 (LoRA Fine-tuning)
- Training completed in ~4.3 minutes  
- Final Loss ≈ 0.26  
- Improved flower structure, color realism, prompt alignment  
- **Key Insight:** Only 0.05% parameters trained → strong domain adaptation  

### 🔹 Task 4 (EDA)
- Dataset has high resolution variance  
- Slight class imbalance (sampled subset)  
- Text descriptions are short, informative, suitable for prompt conditioning  

### 🔹 Task 5 (CLIP)
- Embeddings capture semantic similarity  
- Similar prompts → high cosine similarity  
- Stable embedding norms (~28)  

### 🔹 Task 6 (CGAN)
- Generator learns shape-specific structure  
- Clear separation between Circle, Square, Triangle  
- Losses converge, visual quality improves across epochs  

## 📁 Outputs
- **Task 3**  
- **Task 4**  
- **Task 5:**   
- **Task 6:**  
---

## 🚀 Key Takeaways
- **LoRA** enables efficient fine-tuning of massive diffusion models  
- **Text + Image alignment (CLIP)** is crucial for generative tasks  
- **EDA** is essential before training generative models  
- **GANs vs Diffusion models:**  
  - GAN → fast & sharp  
  - Diffusion → stable & high-quality  
