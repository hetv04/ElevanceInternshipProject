#task2 attention enhanced GAN

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset

class SelfAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # q,k,v
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels,      kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        N = H * W  
        Q = self.query(x).view(B, -1, N).permute(0, 2, 1) 
        K = self.key(x).view(B, -1, N)                     
        V = self.value(x).view(B, -1, N)                    
        attn = self.softmax(torch.bmm(Q, K))                
        out = torch.bmm(V, attn.permute(0, 2, 1))         
        out = out.view(B, C, H, W)
        return self.gamma * out + x
    
class CrossAttention(nn.Module):
    def __init__(self, img_channels: int, text_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = img_channels // num_heads
        self.q_proj = nn.Conv2d(img_channels, img_channels, kernel_size=1)
        self.k_proj = nn.Linear(text_dim, img_channels)
        self.v_proj = nn.Linear(text_dim, img_channels)
        self.out_proj = nn.Conv2d(img_channels, img_channels, kernel_size=1)
        self.scale = self.head_dim ** -0.5
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, img_feat: torch.Tensor, text_emb: torch.Tensor):
        B, C, H, W = img_feat.shape
        N = H * W
        Q = self.q_proj(img_feat).view(B, C, N).permute(0, 2, 1)  
        K = self.k_proj(text_emb).unsqueeze(1)  
        V = self.v_proj(text_emb).unsqueeze(1)  
        attn = torch.softmax(torch.bmm(Q, K.permute(0, 2, 1)) * self.scale, dim=-1) 
        out = torch.bmm(attn, V)                  # (B, N, C)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        out = self.out_proj(out)
        return self.gamma * out + img_feat


# generator
class AttentionGenerator(nn.Module):
    def __init__(self, noise_dim: int = 100, text_dim: int = 512, img_channels: int = 3):
        super().__init__()
        input_dim = noise_dim + text_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512 * 4 * 4),
            nn.ReLU(True)
        )

        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.self_attn = SelfAttention(128)
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.cross_attn = CrossAttention(img_channels=64, text_dim=text_dim, num_heads=4)
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, noise: torch.Tensor, text_emb: torch.Tensor):
        x = torch.cat([noise, text_emb], dim=1)
        x = self.fc(x).view(-1, 512, 4, 4)
        x = self.block1(x)                     
        x = self.block2(x)                   
        x = self.self_attn(x)                 
        x = self.block3(x)               
        x = self.cross_attn(x, text_emb)    
        x = self.block4(x)                  
        return x
    
#discriminator
class AttentionDiscriminator(nn.Module):
    def __init__(self, text_dim: int = 512, img_channels: int = 3):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, 64 * 64)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(img_channels + 1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor, text_emb: torch.Tensor):
        text_map = self.text_proj(text_emb).view(-1, 1, 64, 64)
        x = torch.cat([img, text_map], dim=1)
        return self.fc(self.conv_layers(x))
    
#train,compare
class AttentionGANTrainer:
    def __init__(self, noise_dim: int = 100, text_dim: int = 512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_dim = noise_dim
        print(f"Attention GAN on {self.device}")
        self.G = AttentionGenerator(noise_dim, text_dim).to(self.device)
        self.D = AttentionDiscriminator(text_dim).to(self.device)
        self.opt_G = optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()
        self.g_losses = []
        self.d_losses = []

    def train(self, text_embeddings: torch.Tensor, epochs: int = 50, batch_size: int = 16):
        print(f"\nTraining Attention GAN: {epochs} epochs")

        N = max(len(text_embeddings), 64)
        real_images = torch.randn(N, 3, 64, 64).clamp(-1, 1)
        repeated_emb = text_embeddings.repeat((N // len(text_embeddings)) + 1, 1)[:N]
        dataset = TensorDataset(real_images, repeated_emb)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            epoch_d, epoch_g = 0, 0

            for real_imgs, emb in loader:
                real_imgs = real_imgs.to(self.device)
                emb = emb.to(self.device)
                bs = real_imgs.size(0)

                real_labels = torch.ones(bs, 1).to(self.device)
                fake_labels = torch.zeros(bs, 1).to(self.device)

                # dis
                self.opt_D.zero_grad()
                d_loss = (
                    self.criterion(self.D(real_imgs, emb), real_labels) +
                    self.criterion(self.D(self.G(torch.randn(bs, self.noise_dim).to(self.device), emb).detach(), emb), fake_labels)
                ) / 2
                d_loss.backward()
                self.opt_D.step()

                # gen
                self.opt_G.zero_grad()
                fake = self.G(torch.randn(bs, self.noise_dim).to(self.device), emb)
                g_loss = self.criterion(self.D(fake, emb), real_labels)
                g_loss.backward()
                self.opt_G.step()
                epoch_d += d_loss.item()
                epoch_g += g_loss.item()
            self.d_losses.append(epoch_d / len(loader))
            self.g_losses.append(epoch_g / len(loader))
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:3d}/{epochs}] | D Loss: {self.d_losses[-1]:.4f} | G Loss: {self.g_losses[-1]:.4f}")
        print("done")

    def generate(self, text_emb: torch.Tensor, n: int = 4):
        self.G.eval()
        with torch.no_grad():
            noise = torch.randn(n, self.noise_dim).to(self.device)
            emb = text_emb.unsqueeze(0).expand(n, -1).to(self.device)
            imgs = self.G(noise, emb)
            imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
        return [imgs[i].cpu().permute(1, 2, 0).numpy() for i in range(n)]

#plo
def visualize_attention_map(generator: AttentionGenerator, noise: torch.Tensor, text_emb: torch.Tensor):

    device = next(generator.parameters()).device
    generator.eval()
    attention_output = {}
    def hook_fn(module, input, output):
        attention_output['map'] = output.detach()
    hook = generator.self_attn.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = generator(noise.to(device), text_emb.to(device))
    hook.remove()
    attn_feat = attention_output['map'][0]      
    attn_mean = attn_feat.mean(dim=0).cpu().numpy()  
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Attention Mechanisms in Generator", fontsize=13, fontweight='bold')
    im = axes[0].imshow(attn_mean, cmap='hot')
    axes[0].set_title(f"Self-Attention Feature Map\n(16×16, mean across channels)")
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0])
    gammas = {
        'Self-Attention\nγ': generator.self_attn.gamma.item(),
        'Cross-Attention\nγ': generator.cross_attn.gamma.item()
    }
    axes[1].bar(gammas.keys(), gammas.values(), color=['steelblue', 'coral'])
    axes[1].set_title("Learnable Gamma Values\n(how much attention contributes)")
    axes[1].set_ylabel("γ value")
    axes[1].axhline(0, color='black', linewidth=0.8)
    generator.eval()
    with torch.no_grad():
        out = generator(noise[:1].to(device), text_emb[:1].to(device))
        out = (out * 0.5 + 0.5).clamp(0, 1)
        img = out[0].cpu().permute(1, 2, 0).numpy()
    axes[2].imshow(img)
    axes[2].set_title("Sample Generated Image\n(Attention GAN)")
    axes[2].axis('off')

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/task2_attention_maps.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: outputs/task2_attention_maps.png")

def compare_losses(baseline_losses: dict, attention_losses: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Baseline GAN vs Attention GAN — Training Comparison", fontsize=13, fontweight='bold')
    for ax, key, title in zip(axes, ['g', 'd'], ['Generator Loss', 'Discriminator Loss']):
        ax.plot(baseline_losses[key], label='Baseline GAN', color='gray', linestyle='--')
        ax.plot(attention_losses[key], label='Attention GAN', color='steelblue')
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/task2_loss_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: outputs/task2_loss_comparison.png")


def compare_generated_images(baseline_gan, attention_trainer, text_embeddings, prompts):
    """Side-by-side visual comparison of outputs."""
    n_prompts = min(3, len(prompts))
    fig, axes = plt.subplots(n_prompts, 8, figsize=(18, 3 * n_prompts))
    fig.suptitle("Generated Images: Baseline (left 4) vs Attention GAN (right 4)", fontsize=12, fontweight='bold')

    for i in range(n_prompts):
        emb = text_embeddings[i]

        # Baseline samples (reuse Task 1 GAN)
        baseline_samples = baseline_gan.generate_sample(emb, n=4)
        # Attention samples
        attention_samples = attention_trainer.generate(emb, n=4)

        for j in range(4):
            axes[i][j].imshow(baseline_samples[j])
            axes[i][j].axis('off')
            if j == 0:
                axes[i][j].set_title(f"Base\n{prompts[i][:15]}...", fontsize=7)

        for j in range(4):
            axes[i][j + 4].imshow(attention_samples[j])
            axes[i][j + 4].axis('off')
            if j == 0:
                axes[i][j + 4].set_title(f"Attn\n{prompts[i][:15]}...", fontsize=7)

    plt.tight_layout()
    plt.savefig("outputs/task2_image_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: outputs/task2_image_comparison.png")


# ─────────────────────────────────────────────
# SECTION 6: Run Task 2
# ─────────────────────────────────────────────

def run_task2():
    print("=" * 60)
    print("  TASK 2: Attention-Enhanced GAN")
    print("=" * 60)

    # Reuse preprocessing + embeddings from Task 1
    preprocessor = TextPreprocessor()
    prompts = [
        "A beautiful mountain landscape at sunrise, highly detailed",
        "Portrait of a cyberpunk warrior, neon lights, digital art",
        "Cute cartoon cat wearing a wizard hat, kawaii style",
        "Abstract geometric pattern in vivid colors, modern art",
        "Ancient ruins in a dense jungle, photorealistic"
    ]
    cleaned = preprocessor.batch_preprocess(prompts)

    extractor = TextEmbeddingExtractor()
    embeddings = extractor.get_embeddings(cleaned)
    print(f"Embeddings ready: {embeddings.shape}")

    # Train Baseline GAN (from Task 1) for comparison
    print("\n[1/3] Training Baseline GAN (for comparison)...")
    baseline = TextToImageGAN(noise_dim=100, text_dim=512) # reuse Task 1 class
    baseline = TextToImageGAN(noise_dim=100, text_dim=512)
    baseline.train(embeddings, epochs=50, batch_size=16)

    # Train Attention GAN
    print("\n[2/3] Training Attention GAN...")
    attn_trainer = AttentionGANTrainer(noise_dim=100, text_dim=512)
    attn_trainer.train(embeddings, epochs=50, batch_size=16)

    # Visualizations
    print("\n[3/3] Generating Visualizations...")

    # Attention maps
    noise_sample = torch.randn(4, 100)
    visualize_attention_map(attn_trainer.G, noise_sample, embeddings[:4])

    # Loss comparison
    compare_losses(
        baseline_losses={'g': baseline.g_losses, 'd': baseline.d_losses},
        attention_losses={'g': attn_trainer.g_losses, 'd': attn_trainer.d_losses}
    )

    # Image comparison
    compare_generated_images(baseline, attn_trainer, embeddings, cleaned)

    print("\n" + "=" * 60)
    print("  Task 2 Complete! Outputs saved to outputs/ folder.")
    print("=" * 60)


run_task2()