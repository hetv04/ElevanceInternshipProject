import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
import time
from PIL import Image
from tqdm.auto import tqdm

from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel
)
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model

#flower desc

FLOWER_DESCRIPTIONS = {
    0:  ("pink primrose",         "A delicate pink primrose with soft rounded petals arranged symmetrically around a yellow center."),
    1:  ("hard-leaved pocket orchid", "A rare orchid with waxy, hard leaves and intricate patterned flowers in purple and white."),
    2:  ("canterbury bells",      "Bell-shaped flowers in shades of blue, purple, and white that droop elegantly from tall stems."),
    3:  ("sweet pea",             "Fragrant climbing flowers with ruffled petals in pink, purple, red, and white hues."),
    4:  ("english marigold",      "Bright orange and yellow flowers with dense layered petals, known for their strong scent."),
    5:  ("tiger lily",            "Bold orange flowers with dark spots and recurved petals, resembling a tiger's markings."),
    6:  ("moon orchid",           "Pure white orchid with a purple and yellow center, one of the most popular orchid species."),
    7:  ("bird of paradise",      "Striking tropical flower with orange and blue petals resembling a bird in flight."),
    8:  ("monkshood",             "Deep purple hood-shaped flowers on tall spikes, named for their monk's cowl resemblance."),
    9:  ("globe thistle",         "Spherical spiky blue-purple flower heads on silvery stems, loved by bees and butterflies."),
    10: ("snapdragon",            "Colorful flowers that open and close like a mouth when squeezed, in a wide range of colors."),
    11: ("colt's foot",           "Bright yellow daisy-like flowers that bloom before their leaves appear in early spring."),
    12: ("king protea",           "South Africa's national flower with a large bowl-shaped bloom of pink and white petals."),
    13: ("spear thistle",         "Tall prickly plant with a vivid purple flower head surrounded by sharp spiny bracts."),
    14: ("yellow iris",           "Elegant yellow iris growing near water with sword-like leaves and golden-yellow petals."),
    15: ("globe-flower",          "Pale yellow globe-shaped flowers that resemble large buttercups, growing in damp meadows."),
    16: ("purple coneflower",     "Daisy-like purple flowers with a prominent spiky orange-brown center cone."),
    17: ("peruvian lily",         "Colorful lily-like flowers with streaked petals in shades of yellow, orange, pink and red."),
    18: ("balloon flower",        "Puffed up balloon-like buds that open into star-shaped blue or white flowers."),
    19: ("giant white arum lily", "Large white funnel-shaped spathe surrounding a yellow spike, grown near water."),
    20: ("fire lily",             "Vivid red-orange lily flowers on tall stems, native to South Africa."),
    21: ("pincushion flower",     "Rounded flower heads in blue, purple, or pink with prominent stamens like pins in a cushion."),
    22: ("fritillary",            "Bell-shaped nodding flowers with a distinctive checkered pattern in purple and white."),
    23: ("red ginger",            "Exotic tropical flower with waxy red bracts stacked in a cone-like formation."),
    24: ("grape hyacinth",        "Dense spikes of tiny deep blue-purple urn-shaped flowers resembling a bunch of grapes."),
    25: ("corn poppy",            "Vivid scarlet red flowers with black centers and delicate paper-thin petals."),
    26: ("prince of wales feathers", "Feathery plumes of creamy white flowers arching gracefully from tall stems."),
    27: ("stemless gentian",      "Deep blue trumpet-shaped flowers growing close to the ground in alpine meadows."),
    28: ("artichoke",             "Large silvery-green flower bud with overlapping scales that opens into a purple bloom."),
    29: ("sweet william",         "Clusters of small fragrant flowers in red, pink, white or bicolored patterns."),
    30: ("carnation",             "Fringed petals in a wide array of colors with a spicy clove-like fragrance."),
    31: ("garden phlox",          "Large fragrant clusters of star-shaped flowers in pink, purple, white or red."),
    32: ("love in the mist",      "Pale blue flowers surrounded by a delicate lacy green mist of finely cut leaves."),
    33: ("mexican aster",         "Daisy-like flowers in pink, purple and white with bright yellow centers."),
    34: ("alpine sea holly",      "Steel-blue spiky bracts surrounding a central cone, adapted to rocky alpine terrain."),
    35: ("ruby-lipped cattleya",  "Showy orchid with large lavender petals and a rich magenta-purple frilled lip."),
    36: ("cape flower",           "Bright daisy-like flowers in vivid orange and yellow from South Africa."),
    37: ("great masterwort",      "Star-shaped flower heads with papery white and pink bracts and tiny central florets."),
    38: ("siam tulip",            "Pink tulip-like tropical flower with a prominent purple-tipped pink bract."),
    39: ("lenten rose",           "Nodding cup-shaped flowers in white, pink, purple or near-black in late winter."),
    40: ("barbeton daisy",        "Bold flowers in red, orange, yellow or pink with a prominent dark center disc."),
    41: ("daffodil",              "Cheerful yellow or white flowers with a central trumpet-shaped corona."),
    42: ("sword lily",            "Tall spikes of funnel-shaped flowers in red, pink, yellow, white or orange."),
    43: ("poinsettia",            "Bright red star-shaped leaf bracts surrounding small yellow flowers, iconic at Christmas."),
    44: ("bolero deep blue",      "Deep intense blue flowers with a velvety texture and contrasting yellow stamens."),
    45: ("wallflower",            "Fragrant four-petaled flowers in warm shades of orange, red, yellow and brown."),
    46: ("marigold",              "Bright orange and yellow pompom flowers with a pungent earthy scent."),
    47: ("buttercup",             "Tiny glossy yellow cup-shaped flowers growing in meadows and along roadsides."),
    48: ("oxeye daisy",           "Classic white daisy with a large yellow center disc and long white ray petals."),
    49: ("common dandelion",      "Bright yellow composite flowers followed by spherical white seed clocks."),
    50: ("petunia",               "Trumpet-shaped flowers in a huge range of colors, solids, stripes and patterns."),
    51: ("wild pansy",            "Small tricolored flowers in purple, yellow and white with distinctive face-like markings."),
    52: ("primula",               "Cheerful clusters of five-petaled flowers in yellow, pink, red, purple or white."),
    53: ("sunflower",             "Large circular flowers with bright yellow ray petals surrounding a brown seed-packed center."),
    54: ("pelargonium",           "Rounded clusters of small flowers in red, pink, white or salmon above lobed leaves."),
    55: ("bishop of llandaff",    "Fiery red semi-double flowers with dark bronze-black foliage, a dramatic dahlia variety."),
    56: ("gaura",                 "Delicate white or pink butterfly-like flowers dancing on tall wiry stems."),
    57: ("geranium",              "Five-petaled flowers in pink, purple, blue or white with attractive lobed leaves."),
    58: ("orange dahlia",         "Bold orange ball-shaped dahlia with symmetrically arranged petals."),
    59: ("pink-yellow dahlia",    "Bicolored dahlia with soft pink outer petals fading to yellow at the center."),
    60: ("cautleya spicata",      "Yellow flowers emerging from deep red bracts on upright spikes in shaded gardens."),
    61: ("japanese anemone",      "Elegant white or pink saucer-shaped flowers with golden yellow stamens in autumn."),
    62: ("black-eyed susan",      "Golden-yellow daisy flowers with a prominent dark brown-black central cone."),
    63: ("silverbush",            "Small white flowers covering silvery-leaved shrubs in a dense carpet."),
    64: ("californian poppy",     "Silky orange or yellow cup-shaped flowers that close at night and in cloudy weather."),
    65: ("osteospermum",          "African daisy with white, pink, purple or yellow petals and a blue-purple center."),
    66: ("spring crocus",         "Small goblet-shaped flowers in purple, white or yellow emerging through snow in spring."),
    67: ("bearded iris",          "Large showy flowers with falls, standards and a fuzzy beard in countless color combinations."),
    68: ("windflower",            "Delicate cup-shaped flowers in white, pink or purple swaying on slender stems."),
    69: ("tree poppy",            "Large creamy white flowers with a golden center on a tall shrubby plant."),
    70: ("gazania",               "Striking daisy flowers with brilliant orange, yellow or red petals with dark stripe patterns."),
    71: ("azalea",                "Masses of funnel-shaped flowers in pink, red, white, orange or purple covering the shrub."),
    72: ("water lily",            "Floating aquatic flowers in white, pink, yellow or purple with waxy rounded leaves."),
    73: ("rose",                  "Classic multi-petaled flowers in every color with a sweet fragrance, symbol of love."),
    74: ("thorn apple",           "Large white or purple trumpet-shaped flowers on a coarse branching plant."),
    75: ("morning glory",         "Trumpet-shaped flowers in vivid blue, purple or pink that open in the morning sun."),
    76: ("passion flower",        "Exotic intricate flowers with a crown of filaments in purple, blue and white."),
    77: ("lotus",                 "Sacred aquatic flower with large pink or white petals rising above muddy water."),
    78: ("toad lily",             "Orchid-like spotted flowers in white and purple growing in shaded woodland gardens."),
    79: ("anthurium",             "Glossy heart-shaped red spathe with a yellow spadix, a popular tropical houseplant."),
    80: ("frangipani",            "Fragrant white and yellow flowers with five waxy petals, used in Hawaiian leis."),
    81: ("clematis",              "Climbing vine with star-shaped flowers in purple, pink, white or red."),
    82: ("hibiscus",              "Large trumpet-shaped flowers in red, pink, yellow or white with a prominent stamen column."),
    83: ("columbine",             "Nodding spurred flowers in blue, purple, yellow or red with a delicate airy appearance."),
    84: ("desert-rose",           "Pink trumpet-shaped flowers on a succulent with a swollen water-storing trunk."),
    85: ("tree mallow",           "Tall shrub with large pink or purple hollyhock-like flowers on woody stems."),
    86: ("magnolia",              "Large goblet-shaped flowers in white, pink or purple appearing before the leaves in spring."),
    87: ("cyclamen",              "Reflexed swept-back petals in pink, red or white above beautifully marbled leaves."),
    88: ("watercress",            "Tiny white four-petaled flowers on aquatic plants grown in flowing streams."),
    89: ("canna lily",            "Bold tropical flowers in red, orange or yellow on tall stems with broad paddle leaves."),
    90: ("hippeastrum",           "Large trumpet-shaped flowers in red, pink, white or striped patterns on hollow stems."),
    91: ("bee balm",              "Shaggy scarlet or purple flower heads on tall stems, highly attractive to bees."),
    92: ("ball moss",             "Tiny violet flowers on a grey-green epiphytic plant growing on tree branches."),
    93: ("foxglove",              "Tall spikes of tubular flowers in pink, purple or white with spotted throats."),
    94: ("bougainvillea",         "Vivid paper-thin bracts in magenta, orange, pink or white surrounding tiny white flowers."),
    95: ("camellia",              "Perfect rose-like flowers in white, pink or red with glossy dark evergreen leaves."),
    96: ("mallow",                "Saucer-shaped flowers in pink or purple with deeply veined petals and a central column."),
    97: ("mexican petunia",       "Tubular purple flowers on upright stems with narrow dark green leaves."),
    98: ("bromelia",              "Tropical rosette plant with a vivid red and yellow flower spike rising from the center."),
    99: ("blanket flower",        "Daisy-like flowers in fiery red and yellow patterns resembling Native American blankets."),
    100:("trumpet creeper",       "Clusters of orange-red trumpet-shaped flowers on a vigorous woody climbing vine."),
    101:("blackberry lily",       "Orange flowers with dark spots followed by clusters of shiny black seeds."),
}

class SDFlowerDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer, size: int = 512):
        self.tokenizer = tokenizer
        self.size = size

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            # SD VAE expects [-1, 1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.dataset = torchvision.datasets.Flowers102(
            root=data_dir,
            split="train",
            transform=self.transform,
            download=True
        )
        print(f"dataset ready: {len(self.dataset)} flower images")
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # Get text description for this flower class
        name, desc = FLOWER_DESCRIPTIONS.get(
            label,
            (f"flower class {label}", f"a beautiful flower of class {label}")
        )
        caption = f"a photo of a {name}, {desc[:60]}"
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        return {
            "pixel_values": image,
            "input_ids": tokens.input_ids.squeeze(0),
            "label": label,
            "caption": caption
        }

#lora config, modell
def load_sd_with_lora(model_id: str = "runwayml/stable-diffusion-v1-5"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SD 1.5 on {device}...")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device)
    text_encoder.requires_grad_(False) 
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float16
    ).to(device)
    vae.requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float32
    ).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    lora_config = LoraConfig(
        r=4,                         
        lora_alpha=4,               
        target_modules=[           
            "to_q", "to_v",     n
        ],
        lora_dropout=0.1,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in unet.parameters())
    print(f"\nLoRA Applied to UNet:")
    print(f"  Trainable params : {trainable:,}  ({100*trainable/total:.2f}% of total)")
    print(f"  Frozen params    : {total - trainable:,}")
    print(f"  Total params     : {total:,}")
    return tokenizer, text_encoder, vae, unet, noise_scheduler, device

#train loop

def train_lora(
    tokenizer, text_encoder, vae, unet,
    noise_scheduler, device, dataloader,
    num_steps: int = 300,
    lr: float = 1e-4,
    gradient_accumulation: int = 4
):
    """
    Fine-tune LoRA weights on flower images.
    Uses gradient accumulation to simulate larger batch size.
    """
    optimizer = AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-2
    )

    losses = []
    step = 0
    unet.train()

    print(f"\nStarting LoRA Fine-tuning:")
    print(f"  Steps              : {num_steps}")
    print(f"  Learning Rate      : {lr}")
    print(f"  Gradient Accum     : {gradient_accumulation}")
    print(f"  Effective Batch    : {dataloader.batch_size * gradient_accumulation}\n")

    start_time = time.time()
    optimizer.zero_grad()
    data_iter = iter(dataloader)
    pbar = tqdm(total=num_steps, desc="Fine-tuning SD with LoRA")

    while step < num_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        pixel_values = batch["pixel_values"].to(device)
        input_ids    = batch["input_ids"].to(device)
        with torch.no_grad():
            latents = vae.encode(
                pixel_values.to(dtype=torch.float16)
            ).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            latents = latents.to(dtype=torch.float32)
        noise     = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=device
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        with torch.no_grad():
            encoder_hidden_states = text_encoder(input_ids)[0].to(dtype=torch.float32)
        noise_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states
        ).sample
        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        loss = loss / gradient_accumulation
        loss.backward()
        losses.append(loss.item() * gradient_accumulation)
        if (step + 1) % gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in unet.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            optimizer.zero_grad()
        step += 1
        pbar.update(1)
        if step % 50 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / step * (num_steps - step)
            pbar.set_postfix({
                'loss': f'{np.mean(losses[-20:]):.4f}',
                'ETA': f'{eta/60:.1f}min'
            })
        if step % 100 == 0: #clean memory
            gc.collect()
            torch.cuda.empty_cache()

    pbar.close()
    print(f"\nFine-tuning complete in {(time.time()-start_time)/60:.1f} minutes")
    return losses


#gen image
def generate_images_for_comparison(
    model_id: str,
    unet_finetuned,
    device,
    prompts: list,
    seed: int = 42
):
    print("Loading pipeline for image generation...")
    pipe_original = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    pipe_original.enable_attention_slicing()

    original_images = []
    print("Generating with original SD 1.5...")
    for prompt in prompts:
        generator = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            img = pipe_original(
                prompt,
                num_inference_steps=25,
                guidance_scale=7.5,
                generator=generator,
                height=512, width=512
            ).images[0]
        original_images.append(img)
    del pipe_original
    gc.collect()
    torch.cuda.empty_cache()
    pipe_finetuned = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    unet_finetuned.eval()
    unet_merged = unet_finetuned.merge_adapter()
    pipe_finetuned.unet = unet_finetuned.to(dtype=torch.float16)
    pipe_finetuned.enable_attention_slicing()

    finetuned_images = []
    print("Generating with LoRA fine-tuned SD 1.5...")
    for prompt in prompts:
        generator = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            img = pipe_finetuned(
                prompt,
                num_inference_steps=25,
                guidance_scale=7.5,
                generator=generator,
                height=512, width=512
            ).images[0]
        finetuned_images.append(img)

    del pipe_finetuned
    gc.collect()
    torch.cuda.empty_cache()

    return original_images, finetuned_images

#plot

def plot_lora_losses(losses: list):
    window = 20
    smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(losses, alpha=0.3, color='steelblue', label='Raw loss')
    ax.plot(range(window-1, len(losses)), smoothed,
            color='steelblue', linewidth=2, label=f'Smoothed (window={window})')
    ax.set_title("LoRA Fine-tuning Loss — Stable Diffusion 1.5 on Oxford Flowers",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("Training Step")
    ax.set_ylabel("MSE Noise Prediction Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/task3_lora_loss.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: outputs/task3_lora_loss.png")

def plot_before_after(original_images: list, finetuned_images: list, prompts: list):
    """Side-by-side before/after comparison."""
    n = len(prompts)
    fig, axes = plt.subplots(n, 2, figsize=(12, 5 * n))
    fig.suptitle(
        "Stable Diffusion 1.5: Before vs After LoRA Fine-tuning on Oxford Flowers",
        fontsize=13, fontweight='bold'
    )
    for i, (orig, fine, prompt) in enumerate(zip(original_images, finetuned_images, prompts)):
        axes[i][0].imshow(np.array(orig))
        axes[i][0].set_title(f"BEFORE (Original SD 1.5)\n\"{prompt[:50]}\"",
                              fontsize=8, color='gray')
        axes[i][0].axis('off')
        axes[i][1].imshow(np.array(fine))
        axes[i][1].set_title(f"AFTER (LoRA Fine-tuned)\n\"{prompt[:50]}\"",
                              fontsize=8, color='steelblue')
        axes[i][1].axis('off')
    plt.tight_layout()
    plt.savefig("outputs/task3_before_after.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: outputs/task3_before_after.png")
def plot_lora_architecture():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    fig.patch.set_facecolor('#f8f9fa')

    ax.text(0.5, 0.95, "How LoRA Fine-tunes Stable Diffusion",
            ha='center', va='top', fontsize=14, fontweight='bold', transform=ax.transAxes)

    info = [
        ("Original SD 1.5 UNet", "860M parameters\nAll frozen (not trained)\nPre-trained on LAION-5B"),
        ("LoRA Adapter (r=4)", "~1M trainable parameters\nAdded to: to_q, to_v layers\nOnly 0.1% of total params"),
        ("Fine-tuned on Flowers", "300 steps on Oxford 102\nLearns flower-specific features\nDomain adaptation"),
        ("Result", "SD generates better flowers\nOriginal knowledge preserved\nFast & memory efficient"),
    ]
    colors = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6']
    for idx, (title, body) in enumerate(info):
        x = 0.05 + idx * 0.24
        rect = plt.Rectangle((x, 0.15), 0.21, 0.65,
                               transform=ax.transAxes,
                               facecolor=colors[idx], alpha=0.15,
                               edgecolor=colors[idx], linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 0.105, 0.75, title, ha='center', va='center',
                fontsize=9, fontweight='bold', transform=ax.transAxes,
                color=colors[idx])
        ax.text(x + 0.105, 0.42, body, ha='center', va='center',
                fontsize=8, transform=ax.transAxes, color='#2c3e50')
        if idx < 3:
            ax.annotate("", xy=(x + 0.24, 0.48), xytext=(x + 0.21, 0.48),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle="->", color='#7f8c8d', lw=2))

    plt.tight_layout()
    plt.savefig("outputs/task3_lora_architecture.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: outputs/task3_lora_architecture.png")


# lora weights

def save_lora_weights(unet, path: str = "outputs/task3_lora_weights"):
    """Save only the LoRA weights (tiny file ~4MB vs 3.4GB for full model)."""
    os.makedirs(path, exist_ok=True)
    unet.save_pretrained(path)
    print(f"LoRA weights saved to: {path}")
    for f in os.listdir(path):
        size = os.path.getsize(os.path.join(path, f)) / 1024 / 1024
        print(f"  {f}: {size:.2f} MB")

def run_task3():
    print("=" * 60)
    print("  TASK 3: LoRA Fine-tuning of Stable Diffusion 1.5")
    print("  Dataset: Oxford 102 Flowers")
    print("=" * 60)

    os.makedirs("outputs", exist_ok=True)
    MODEL_ID = "runwayml/stable-diffusion-v1-5"
    print("\n[1/5] Loading Stable Diffusion 1.5 with Lora")
    tokenizer, text_encoder, vae, unet, noise_scheduler, device = load_sd_with_lora(MODEL_ID)
    print("\n[2/5] Preparing flower dataset...")
    flower_dataset = SDFlowerDataset(
        data_dir="flowers_data",
        tokenizer=tokenizer,
        size=512
    )
    dataloader = DataLoader(
        flower_dataset,
        batch_size=1,       
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    print("\n[3/5] Plotting LoRA architecture")
    plot_lora_architecture()
    print("\n[4/5] Fine-tuning with LoRA")
    losses = train_lora(
        tokenizer, text_encoder, vae, unet,
        noise_scheduler, device, dataloader,
        num_steps=300,
        lr=1e-4,
        gradient_accumulation=4
    )
    plot_lora_losses(losses)
    print("\n[5/5] Generating before/after comparison image")
    test_prompts = [
        "a photo of a sunflower, bright yellow petals, detailed",
        "a photo of a rose, red petals, garden, detailed",
        "a photo of a lotus, pink flower, water, detailed",
    ]
    original_imgs, finetuned_imgs = generate_images_for_comparison(
        MODEL_ID, unet, device, test_prompts, seed=42
    )
    plot_before_after(original_imgs, finetuned_imgs, test_prompts)
    save_lora_weights(unet)
    print("  Task 3 Complete! Output files:")
    print("  - task3_lora_architecture.png")
    print("  - task3_lora_loss.png")
    print("  - task3_before_after.png")
    print("  - task3_lora_weights/ (saved LoRA adapter)")

run_task3()