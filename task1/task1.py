#Task 1:Text-to-Image Pipeline

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import re
import os
import time
from datetime import datetime
from transformers import CLIPTokenizer, CLIPTextModel

# Text Preprocessing

class TextPreprocessor:

    def __init__(self):
        # Common filler words remove
        self.stopwords = {"a", "an", "the", "of", "in", "on", "at", "is", "with"}
    def clean(self, text: str) -> str:
        """Lowercase, strip punctuation, normalize whitespace."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s,]", "", text) 
        text = re.sub(r"\s+", " ", text)
        return text
    def remove_stopwords(self, text: str) -> str:
        """Remove basic stopwords (optional step)."""
        tokens = text.split()
        filtered = [t for t in tokens if t not in self.stopwords]
        return " ".join(filtered)

    def preprocess(self, text: str, remove_stops: bool = False) -> str:
        """Full preprocessing pipeline."""
        text = self.clean(text)
        if remove_stops:
            text = self.remove_stopwords(text)
        return text

    def batch_preprocess(self, texts: list, remove_stops: bool = False) -> list:
        """Process a list of prompts."""
        return [self.preprocess(t, remove_stops) for t in texts]

#Text embedding with clip
class TextEmbeddingExtractor:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"Clip loaded: {self.device}")
    def get_embeddings(self, texts: list) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        #1 vec/prompt
        embeddings = outputs.pooler_output  # shape N,512
        return embeddings

    def visualize_embeddings(self, texts: list, embeddings: torch.Tensor):
        emb_np = embeddings.cpu().numpy()
        norms = np.linalg.norm(emb_np, axis=1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Clip txt embeddings analysis", fontsize=14, fontweight='bold')
        axes[0].imshow(emb_np[:, :50], aspect='auto', cmap='coolwarm')
        axes[0].set_title("Embedding Heatmap (first 50 dims)")
        axes[0].set_yticks(range(len(texts)))
        axes[0].set_yticklabels([t[:30] + "..." if len(t) > 30 else t for t in texts], fontsize=8)
        axes[0].set_xlabel("Embedding Dimension")
        plt.colorbar(axes[0].images[0], ax=axes[0])

        axes[1].barh(range(len(texts)), norms, color='steelblue')
        axes[1].set_yticks(range(len(texts)))
        axes[1].set_yticklabels([t[:30] + "..." if len(t) > 30 else t for t in texts], fontsize=8)
        axes[1].set_xlabel("L2 Norm")
        axes[1].set_title("Embedding Norms per Prompt")

        plt.tight_layout()
        os.makedirs("outputs", exist_ok=True)
        plt.savefig("outputs/task1_embeddings_analysis.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved: outputs/task1_embeddings_analysis.png")

#Text GAN
class Generator(nn.Module):

    def __init__(self, noise_dim: int = 100, text_dim: int = 512, img_channels: int = 3):
        super().__init__()
        input_dim = noise_dim + text_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512 * 4 * 4),
            nn.ReLU(True),
        )
        self.conv_layers = nn.Sequential(
            # 4x4 → 8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 8x8 → 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 16x16 → 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 32x32 → 64x64
            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Output in [-1, 1]
        )
    def forward(self, noise: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([noise, text_emb], dim=1)
        x = self.net(x)
        x = x.view(-1, 512, 4, 4)
        return self.conv_layers(x)

class Discriminator(nn.Module):
    def __init__(self, text_dim: int = 512, img_channels: int = 3):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, 64 * 64)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(img_channels + 1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        text_map = self.text_proj(text_emb).view(-1, 1, 64, 64)
        x = torch.cat([img, text_map], dim=1)
        x = self.conv_layers(x)
        return self.fc(x)

#train
class TextToImageGAN:
    def __init__(self, noise_dim: int = 100, text_dim: int = 512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_dim = noise_dim
        print(f"start GAN on {self.device}")
        self.G = Generator(noise_dim=noise_dim, text_dim=text_dim).to(self.device)
        self.D = Discriminator(text_dim=text_dim).to(self.device)
        self.opt_G = optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()
        self.g_losses = []
        self.d_losses = []

    def train(self, text_embeddings: torch.Tensor, epochs: int = 50, batch_size: int = 16):
        print(f"\nstart GAN Training: {epochs} epochs, batch size {batch_size}")
        N = max(len(text_embeddings), 64)
        real_images = torch.randn(N, 3, 64, 64).clamp(-1, 1)
        repeated_emb = text_embeddings.repeat(
            (N // len(text_embeddings)) + 1, 1
        )[:N]
        dataset = TensorDataset(real_images, repeated_emb)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            epoch_d_loss = 0
            epoch_g_loss = 0

            for real_imgs, emb in loader:
                real_imgs = real_imgs.to(self.device)
                emb = emb.to(self.device)
                bs = real_imgs.size(0)

                real_labels = torch.ones(bs, 1).to(self.device)
                fake_labels = torch.zeros(bs, 1).to(self.device)
                self.opt_D.zero_grad()
                d_real = self.D(real_imgs, emb)
                d_loss_real = self.criterion(d_real, real_labels)

                noise = torch.randn(bs, self.noise_dim).to(self.device)
                fake_imgs = self.G(noise, emb).detach()
                d_fake = self.D(fake_imgs, emb)
                d_loss_fake = self.criterion(d_fake, fake_labels)

                d_loss = (d_loss_real + d_loss_fake) / 2
                d_loss.backward()
                self.opt_D.step()
                self.opt_G.zero_grad()
                noise = torch.randn(bs, self.noise_dim).to(self.device)
                fake_imgs = self.G(noise, emb)
                g_out = self.D(fake_imgs, emb)
                g_loss = self.criterion(g_out, real_labels)
                g_loss.backward()
                self.opt_G.step()

                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()

            avg_d = epoch_d_loss / len(loader)
            avg_g = epoch_g_loss / len(loader)
            self.d_losses.append(avg_d)
            self.g_losses.append(avg_g)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:3d}/{epochs}] | D Loss: {avg_d:.4f} | G Loss: {avg_g:.4f}")
        print("\ndone")

    def generate_sample(self, text_emb: torch.Tensor, n: int = 4) -> list:
        self.G.eval()
        with torch.no_grad():
            noise = torch.randn(n, self.noise_dim).to(self.device)
            emb = text_emb.unsqueeze(0).expand(n, -1).to(self.device)
            fake = self.G(noise, emb)
            fake = (fake * 0.5 + 0.5).clamp(0, 1)  # [-1,1] → [0,1]
        return [fake[i].cpu().permute(1, 2, 0).numpy() for i in range(n)]

    def plot_losses(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.g_losses, label="Generator Loss", color="blue")
        plt.plot(self.d_losses, label="Discriminator Loss", color="red")
        plt.title("GAN Training Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        os.makedirs("outputs", exist_ok=True)
        plt.savefig("outputs/task1_gan_losses.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved: outputs/task1_gan_losses.png")

    def visualize_generated(self, prompts: list, text_embeddings: torch.Tensor):
        fig, axes = plt.subplots(len(prompts), 4, figsize=(12, 3 * len(prompts)))
        fig.suptitle("GAN Generated Images per Prompt", fontsize=13, fontweight='bold')

        for i, (prompt, emb) in enumerate(zip(prompts, text_embeddings)):
            samples = self.generate_sample(emb, n=4)
            for j, img in enumerate(samples):
                ax = axes[i][j] if len(prompts) > 1 else axes[j]
                ax.imshow(img)
                ax.axis('off')
                if j == 0:
                    ax.set_title(prompt[:25] + "...", fontsize=7)

        plt.tight_layout()
        plt.savefig("outputs/task1_generated_images.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved: outputs/task1_generated_images.png")

#run 
def run_task1_pipeline():
    print("task1")
    print("\n[1/4] Text Preprocessing")
    preprocessor = TextPreprocessor()
    raw_prompts = [
        "A beautiful mountain landscape at sunrise, highly detailed",
        "Portrait of a cyberpunk warrior, neon lights, digital art",
        "Cute cartoon cat wearing a wizard hat, kawaii style",
        "Abstract geometric pattern in vivid colors, modern art",
        "Ancient ruins in a dense jungle, photorealistic"
    ]
    cleaned = preprocessor.batch_preprocess(raw_prompts)
    for raw, clean in zip(raw_prompts, cleaned):
        print(f"  Raw  : {raw}")
        print(f"  Clean: {clean}\n")
    print("[2/4] Generating CLIP Text Embeddings")
    extractor = TextEmbeddingExtractor()
    embeddings = extractor.get_embeddings(cleaned)
    print(f"  Embedding shape: {embeddings.shape}  (N=5 prompts, 512 dims each)")
    extractor.visualize_embeddings(cleaned, embeddings)
    print("\n[3/4] Training Text-Conditioned GAN")
    gan = TextToImageGAN(noise_dim=100, text_dim=512)
    gan.train(embeddings, epochs=50, batch_size=16)
    gan.plot_losses()
    print("\n[4/4] Generating Sample Images from Text Prompts")
    gan.visualize_generated(cleaned, embeddings)

if __name__ == "__main__":
    run_task1_pipeline()