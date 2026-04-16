import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from transformers import CLIPTokenizer, CLIPTextModel

os.makedirs("outputs", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# reusing cached clip from task 3
MODEL_ID = "runwayml/stable-diffusion-v1-5"

print("Load clip")
tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
    MODEL_ID, subfolder="text_encoder",
    torch_dtype=torch.float16
).to(device)
text_encoder.eval()
print("CLIP loaded!")

# sample
prompts = [
    "a red rose in a garden",
    "a sunflower field at sunset",
    "a lotus flower floating on water",
    "a purple orchid with white spots",
    "a yellow daisy in a meadow"
]
print("\nTokenization Results")
tokens = tokenizer(
    prompts,
    padding="max_length",
    max_length=77,
    truncation=True,
    return_tensors="pt"
).to(device)
for i, prompt in enumerate(prompts):
    ids      = tokens["input_ids"][i]
    decoded  = tokenizer.convert_ids_to_tokens(ids)
    mask     = tokens["attention_mask"][i].bool()
    decoded  = [t for t, m in zip(decoded, mask) if m]
    print(f"\nPrompt : {prompt}")
    print(f"Tokens : {decoded}")
    print(f"Length : {mask.sum().item()} tokens")

# generate embeddings
with torch.no_grad():
    output     = text_encoder(**tokens)
    embeddings = output.pooler_output.float() 

print(f"\nEmbedding shape : {embeddings.shape}")
print(f"Each prompt → 512-dimensional vector")
print(f"Min value  : {embeddings.min().item():.4f}")
print(f"Max value  : {embeddings.max().item():.4f}")
print(f"Mean value : {embeddings.mean().item():.4f}")

# preprocess
print("\nPreprocessing Pipeline")
for i, prompt in enumerate(prompts):
    ids  = tokens["input_ids"][i]
    attn = tokens["attention_mask"][i]
    print(f"\n[{i+1}] '{prompt}'")
    print(f"     Input IDs shape     : {ids.shape}")
    print(f"     Attention mask sum  : {attn.sum().item()} active tokens")
    print(f"     Embedding L2 norm   : {embeddings[i].norm().item():.4f}")

# plot
short = [p[:22] + "..." for p in prompts]

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Task 5: Text Preprocessing & Embeddings (CLIP)",
             fontsize=14, fontweight='bold')

im1 = axes[0][0].imshow(
    embeddings.cpu().numpy()[:, :80], cmap='coolwarm', aspect='auto'
)
axes[0][0].set_title("CLIP Embeddings — first 80 of 512 dims")
axes[0][0].set_yticks(range(len(prompts)))
axes[0][0].set_yticklabels(short, fontsize=8)
axes[0][0].set_xlabel("Embedding Dimension")
plt.colorbar(im1, ax=axes[0][0])

norm  = embeddings / embeddings.norm(dim=1, keepdim=True)
sim   = torch.mm(norm, norm.T).cpu().numpy()
im2   = axes[0][1].imshow(sim, cmap='Blues', vmin=0, vmax=1)
axes[0][1].set_title("Cosine Similarity Between Prompts")
axes[0][1].set_xticks(range(len(prompts)))
axes[0][1].set_yticks(range(len(prompts)))
axes[0][1].set_xticklabels(short, rotation=45, ha='right', fontsize=7)
axes[0][1].set_yticklabels(short, fontsize=7)
plt.colorbar(im2, ax=axes[0][1])
for i in range(len(prompts)):
    for j in range(len(prompts)):
        axes[0][1].text(j, i, f"{sim[i,j]:.2f}",
                        ha='center', va='center', fontsize=9)
tok_counts = tokens["attention_mask"].sum(dim=1).cpu().numpy() - 2
axes[1][0].bar(range(len(prompts)), tok_counts, color='steelblue', edgecolor='white')
axes[1][0].set_xticks(range(len(prompts)))
axes[1][0].set_xticklabels(short, rotation=20, ha='right', fontsize=7)
axes[1][0].set_title("Token Count per Prompt")
axes[1][0].set_ylabel("Tokens")
axes[1][0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(tok_counts):
    axes[1][0].text(i, v + 0.1, str(v), ha='center', fontsize=10)

norms = embeddings.norm(dim=1).cpu().numpy()
axes[1][1].barh(range(len(prompts)), norms, color='coral', edgecolor='white')
axes[1][1].set_yticks(range(len(prompts)))
axes[1][1].set_yticklabels(short, fontsize=8)
axes[1][1].set_title("Embedding L2 Norm per Prompt\n(vector magnitude)")
axes[1][1].set_xlabel("L2 Norm")
axes[1][1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig("outputs/task5_embeddings.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: outputs/task5_embeddings.png")
