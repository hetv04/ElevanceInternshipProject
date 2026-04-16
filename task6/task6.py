#task6 conditional gan 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

#synthetic dataset
LABELS     = {"circle": 0, "square": 1, "triangle": 2}
LABEL_NAMES = {v: k for k, v in LABELS.items()}
IMG_SIZE   = 32  
def draw_circle(size=32):
    img = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    r = size // 2 - 4
    for y in range(size):
        for x in range(size):
            if (x - cx)**2 + (y - cy)**2 <= r**2:
                img[y, x] = 1.0
    return img
def draw_square(size=32):
    img = np.zeros((size, size), dtype=np.float32)
    m = 4
    img[m:size-m, m:size-m] = 1.0
    return img
def draw_triangle(size=32):
    img = np.zeros((size, size), dtype=np.float32)
    for y in range(size):
        width = int((y / size) * (size - 8))
        left  = (size - width) // 2
        right = left + width
        if left < right:
            img[y, left:right] = 1.0
    return img

class ShapeDataset(Dataset):
    """Generates circle, square, triangle images with labels."""
    def __init__(self, n_per_class=500):
        self.images = []
        self.labels = []
        drawers = [draw_circle, draw_square, draw_triangle]
        for label, draw_fn in enumerate(drawers):
            base = draw_fn(IMG_SIZE)
            for _ in range(n_per_class):
                noise = np.random.normal(0, 0.05, base.shape).astype(np.float32)
                img   = np.clip(base + noise, 0, 1)
                self.images.append(img)
                self.labels.append(label)
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img   = torch.tensor(self.images[idx]).unsqueeze(0)  
        img   = img * 2 - 1                             
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

# archite cgan
N_CLASSES  = 3
NOISE_DIM  = 64
EMBED_DIM  = 32   

class CGANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(N_CLASSES, EMBED_DIM)

        self.net = nn.Sequential(
            nn.Linear(NOISE_DIM + EMBED_DIM, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),

            nn.Linear(512, IMG_SIZE * IMG_SIZE),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        emb = self.label_emb(labels)               
        x   = torch.cat([noise, emb], dim=1)      
        out = self.net(x)
        return out.view(-1, 1, IMG_SIZE, IMG_SIZE)  


class CGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(N_CLASSES, EMBED_DIM)

        self.net = nn.Sequential(
            nn.Linear(IMG_SIZE * IMG_SIZE + EMBED_DIM, 512),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)      
        emb      = self.label_emb(labels)         
        x        = torch.cat([img_flat, emb], dim=1)
        return self.net(x)

#training
def train_cgan(epochs=100, batch_size=64):
    dataset    = ShapeDataset(n_per_class=500)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    G = CGANGenerator().to(device)
    D = CGANDiscriminator().to(device)

    opt_G     = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D     = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    g_losses, d_losses = [], []

    print(f"Training CGAN: {epochs} epochs")
    for epoch in range(epochs):
        epoch_g, epoch_d = 0, 0

        for real_imgs, labels in dataloader:
            real_imgs = real_imgs.to(device)
            labels    = labels.to(device)
            bs        = real_imgs.size(0)

            real_lbl = torch.ones(bs, 1).to(device)
            fake_lbl = torch.zeros(bs, 1).to(device)

            # Train D
            opt_D.zero_grad()
            noise     = torch.randn(bs, NOISE_DIM).to(device)
            fake_imgs = G(noise, labels).detach()
            d_loss    = (
                criterion(D(real_imgs, labels), real_lbl) +
                criterion(D(fake_imgs, labels), fake_lbl)
            ) / 2
            d_loss.backward()
            opt_D.step()

            # Train G
            opt_G.zero_grad()
            noise     = torch.randn(bs, NOISE_DIM).to(device)
            fake_imgs = G(noise, labels)
            g_loss    = criterion(D(fake_imgs, labels), real_lbl)
            g_loss.backward()
            opt_G.step()

            epoch_d += d_loss.item()
            epoch_g += g_loss.item()

        g_losses.append(epoch_g / len(dataloader))
        d_losses.append(epoch_d / len(dataloader))

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] | D: {d_losses[-1]:.4f} | G: {g_losses[-1]:.4f}")

    print("Training complete!")
    return G, g_losses, d_losses

# ─────────────────────────────────────────────
# SECTION 4: Visualizations
# ─────────────────────────────────────────────

def visualize_generated_shapes(G):
    """Generate shapes for each label and display them."""
    G.eval()
    fig, axes = plt.subplots(3, 8, figsize=(16, 7))
    fig.suptitle("CGAN Generated Shapes from Text Labels",
                 fontsize=14, fontweight='bold')

    with torch.no_grad():
        for row, (label_name, label_id) in enumerate(LABELS.items()):
            labels = torch.full((8,), label_id, dtype=torch.long).to(device)
            noise  = torch.randn(8, NOISE_DIM).to(device)
            imgs   = G(noise, labels)
            imgs   = (imgs * 0.5 + 0.5).clamp(0, 1).cpu()

            for col in range(8):
                axes[row][col].imshow(imgs[col, 0], cmap='gray', vmin=0, vmax=1)
                axes[row][col].axis('off')
                if col == 0:
                    axes[row][col].set_title(f'"{label_name}"', fontsize=10,
                                              fontweight='bold', color='steelblue')

    plt.tight_layout()
    plt.savefig("outputs/task6_generated_shapes.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: outputs/task6_generated_shapes.png")


def visualize_real_vs_generated(G, dataset):
    """Real shapes vs CGAN generated — side by side."""
    G.eval()
    fig, axes = plt.subplots(3, 2, figsize=(6, 9))
    fig.suptitle("Real Shapes vs CGAN Generated", fontsize=13, fontweight='bold')

    with torch.no_grad():
        for row, (label_name, label_id) in enumerate(LABELS.items()):
            # Real
            real_img = draw_circle(IMG_SIZE) if label_id == 0 else \
                       draw_square(IMG_SIZE) if label_id == 1 else \
                       draw_triangle(IMG_SIZE)
            axes[row][0].imshow(real_img, cmap='gray')
            axes[row][0].set_title(f"Real: {label_name}", fontsize=9)
            axes[row][0].axis('off')

            # Generated
            label  = torch.tensor([label_id]).to(device)
            noise  = torch.randn(1, NOISE_DIM).to(device)
            img    = G(noise, label)
            img    = (img * 0.5 + 0.5).clamp(0, 1).cpu().squeeze().numpy()
            axes[row][1].imshow(img, cmap='gray')
            axes[row][1].set_title(f"Generated: \"{label_name}\"", fontsize=9)
            axes[row][1].axis('off')

    plt.tight_layout()
    plt.savefig("outputs/task6_real_vs_generated.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: outputs/task6_real_vs_generated.png")


def plot_cgan_losses(g_losses, d_losses):
    plt.figure(figsize=(10, 4))
    plt.plot(g_losses, label="Generator", color="blue")
    plt.plot(d_losses, label="Discriminator", color="red")
    plt.title("CGAN Training Losses — Shape Generation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/task6_losses.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: outputs/task6_losses.png")


def visualize_label_embeddings(G):
    """Show what each label embedding looks like."""
    fig, ax = plt.subplots(figsize=(8, 3))
    with torch.no_grad():
        all_labels = torch.arange(N_CLASSES).to(device)
        embeddings = G.label_emb(all_labels).cpu().numpy()

    im = ax.imshow(embeddings, cmap='viridis', aspect='auto')
    ax.set_yticks(range(N_CLASSES))
    ax.set_yticklabels(list(LABELS.keys()), fontsize=11)
    ax.set_xlabel("Embedding Dimension")
    ax.set_title("Learned Label Embeddings per Shape Class\n(what the CGAN learned about each label)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("outputs/task6_label_embeddings.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: outputs/task6_label_embeddings.png")


# ─────────────────────────────────────────────
# SECTION 5: Run Task 6
# ─────────────────────────────────────────────

def run_task6():
    print("=" * 60)
    print("  TASK 6: Conditional GAN — Shape Generation")
    print("  Labels: circle, square, triangle")
    print("=" * 60)

    dataset = ShapeDataset(n_per_class=500)
    print(f"Dataset: {len(dataset)} images ({len(dataset)//3} per class)")

    # Train
    G, g_losses, d_losses = train_cgan(epochs=100, batch_size=64)

    # Visualize
    plot_cgan_losses(g_losses, d_losses)
    visualize_generated_shapes(G)
    visualize_real_vs_generated(G, dataset)
    visualize_label_embeddings(G)

    print("\n" + "=" * 60)
    print("  Task 6 Complete! Output files:")
    print("  - task6_losses.png")
    print("  - task6_generated_shapes.png")
    print("  - task6_real_vs_generated.png")
    print("  - task6_label_embeddings.png")
    print("=" * 60)


run_task6()