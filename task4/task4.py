#task 4 examining flower dataset
import warnings
warnings.filterwarnings("ignore")

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from PIL import Image
from collections import Counter

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

# load dataset
def load_dataset_for_analysis(data_dir: str = "flowers_data"):
    """Load at original size for resolution analysis."""
    raw_transform = transforms.Compose([
        transforms.ToTensor() 
    ])
    dataset_raw = torchvision.datasets.Flowers102(
        root=data_dir,
        split="train",
        transform=raw_transform,
        download=True
    )
    return dataset_raw
def load_dataset_resized(data_dir: str = "flowers_data"):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = torchvision.datasets.Flowers102(
        root=data_dir,
        split="train",
        transform=transform,
        download=True
    )
    return dataset


#data stats
def compute_dataset_statistics(dataset_raw, n_samples: int = 500):
    print("\nComputing dataset Stats")

    n_samples = min(n_samples, len(dataset_raw))
    indices = np.random.choice(len(dataset_raw), n_samples, replace=False)

    widths, heights, aspect_ratios = [], [], []
    r_means, g_means, b_means = [], [], []
    class_counts = Counter()

    for i, idx in enumerate(indices):
        img_tensor, label = dataset_raw[int(idx)]
        c, h, w = img_tensor.shape
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)
        class_counts[label] += 1
        r_means.append(img_tensor[0].mean().item())
        g_means.append(img_tensor[1].mean().item())
        b_means.append(img_tensor[2].mean().item())
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_samples} samples...")
    stats = {
        "total_images":    len(dataset_raw),
        "total_classes":   102,
        "sampled":         n_samples,
        "avg_width":       np.mean(widths),
        "avg_height":      np.mean(heights),
        "min_width":       np.min(widths),
        "max_width":       np.max(widths),
        "min_height":      np.min(heights),
        "max_height":      np.max(heights),
        "avg_aspect":      np.mean(aspect_ratios),
        "widths":          widths,
        "heights":         heights,
        "aspect_ratios":   aspect_ratios,
        "r_mean":          np.mean(r_means),
        "g_mean":          np.mean(g_means),
        "b_mean":          np.mean(b_means),
        "r_std":           np.std(r_means),
        "g_std":           np.std(g_means),
        "b_std":           np.std(b_means),
        "class_counts":    class_counts
    }
    print("\n" + "=" * 50)
    print("  DATASET STATISTICS SUMMARY")
    print("=" * 50)
    print(f"  Total Images     : {stats['total_images']}")
    print(f"  Total Classes    : {stats['total_classes']}")
    print(f"  Avg Resolution   : {stats['avg_width']:.0f} x {stats['avg_height']:.0f} px")
    print(f"  Width  Range     : {stats['min_width']} – {stats['max_width']} px")
    print(f"  Height Range     : {stats['min_height']} – {stats['max_height']} px")
    print(f"  Avg Aspect Ratio : {stats['avg_aspect']:.3f}")
    print(f"  RGB Channel Means: R={stats['r_mean']:.3f} G={stats['g_mean']:.3f} B={stats['b_mean']:.3f}")
    print(f"  RGB Channel Stds : R={stats['r_std']:.3f} G={stats['g_std']:.3f} B={stats['b_std']:.3f}")
    print("=" * 50)

    return stats
# eda
def analyze_descriptions():
    descriptions = [v[1] for v in FLOWER_DESCRIPTIONS.values()]
    names        = [v[0] for v in FLOWER_DESCRIPTIONS.values()]

    word_counts = [len(d.split()) for d in descriptions]
    char_counts = [len(d) for d in descriptions]
    all_words = []
    for d in descriptions:
        words = d.lower().replace(",", "").replace(".", "").split()
        all_words.extend(words)

    stop = {"a", "an", "the", "of", "in", "on", "at", "with",
            "and", "or", "to", "from", "by", "for", "its", "that"}
    filtered_words = [w for w in all_words if w not in stop]
    word_freq = Counter(filtered_words).most_common(20)

    print("\n" + "=" * 50)
    print("  DESCRIPTION LENGTH ANALYSIS")
    print("=" * 50)
    print(f"  Total descriptions : {len(descriptions)}")
    print(f"  Avg word count     : {np.mean(word_counts):.1f}")
    print(f"  Min word count     : {np.min(word_counts)} ({names[np.argmin(word_counts)]})")
    print(f"  Max word count     : {np.max(word_counts)} ({names[np.argmax(word_counts)]})")
    print(f"  Avg char count     : {np.mean(char_counts):.1f}")
    print(f"\n  Top 10 most common words:")
    for word, freq in word_freq[:10]:
        print(f"    '{word}': {freq} times")
    print("=" * 50)

    return word_counts, char_counts, word_freq

#plot
def plot_dataset_statistics(stats: dict, word_counts: list,
                             char_counts: list, word_freq: list):
    """4-panel statistics dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Oxford 102 Flowers — Dataset Statistics Dashboard",
                 fontsize=14, fontweight='bold')

    sample_w = stats['widths'][:300]
    sample_h = stats['heights'][:300]
    axes[0][0].scatter(sample_w, sample_h, alpha=0.4, color='steelblue', s=15)
    axes[0][0].axvline(stats['avg_width'],  color='red',   linestyle='--', label=f"Avg W: {stats['avg_width']:.0f}px")
    axes[0][0].axhline(stats['avg_height'], color='green', linestyle='--', label=f"Avg H: {stats['avg_height']:.0f}px")
    axes[0][0].set_title("Image Resolution Distribution (sample of 300)")
    axes[0][0].set_xlabel("Width (px)")
    axes[0][0].set_ylabel("Height (px)")
    axes[0][0].legend(fontsize=8)
    axes[0][0].grid(True, alpha=0.3)

    axes[0][1].hist(word_counts, bins=15, color='coral', edgecolor='white', linewidth=0.5)
    axes[0][1].axvline(np.mean(word_counts), color='darkred', linestyle='--',
                        label=f"Mean: {np.mean(word_counts):.1f} words")
    axes[0][1].set_title("Description Length Distribution (word count)")
    axes[0][1].set_xlabel("Number of Words")
    axes[0][1].set_ylabel("Number of Descriptions")
    axes[0][1].legend(fontsize=8)
    axes[0][1].grid(True, alpha=0.3)

    top_words  = [w for w, _ in word_freq[:15]]
    top_counts = [c for _, c in word_freq[:15]]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 15))
    axes[1][0].barh(top_words[::-1], top_counts[::-1], color=colors)
    axes[1][0].set_title("Top 15 Most Common Words in Descriptions")
    axes[1][0].set_xlabel("Frequency")
    axes[1][0].grid(True, alpha=0.3, axis='x')

    channels = ['Red', 'Green', 'Blue']
    means = [stats['r_mean'], stats['g_mean'], stats['b_mean']]
    stds  = [stats['r_std'],  stats['g_std'],  stats['b_std']]
    colors_rgb = ['#e74c3c', '#2ecc71', '#3498db']
    bars = axes[1][1].bar(channels, means, yerr=stds, color=colors_rgb,
                           capsize=8, edgecolor='white', linewidth=1.5)
    axes[1][1].set_title("Mean Pixel Intensity per RGB Channel\n(±1 std)")
    axes[1][1].set_ylabel("Mean Intensity (0–1)")
    axes[1][1].set_ylim(0, 1)
    axes[1][1].grid(True, alpha=0.3, axis='y')
    for bar, mean in zip(bars, means):
        axes[1][1].text(bar.get_x() + bar.get_width()/2,
                        mean + 0.03, f"{mean:.3f}", ha='center', fontsize=10)

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/task4_statistics_dashboard.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: outputs/task4_statistics_dashboard.png")

def plot_class_distribution(stats: dict, n_show: int = 30):

    class_counts = stats['class_counts']
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:n_show]
    labels  = [FLOWER_DESCRIPTIONS[c][0][:15] for c, _ in sorted_classes]
    counts  = [cnt for _, cnt in sorted_classes]

    fig, ax = plt.subplots(figsize=(16, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, n_show))
    ax.bar(range(n_show), counts, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(n_show))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_title(f"Class Distribution — Top {n_show} Classes by Sample Count",
                 fontsize=13, fontweight='bold')
    ax.set_ylabel("Sample Count")
    ax.axhline(np.mean(counts), color='red', linestyle='--',
               label=f"Mean: {np.mean(counts):.1f}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("outputs/task4_class_distribution.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: outputs/task4_class_distribution.png")

def display_images_with_descriptions(dataset_resized, n_display: int = 12):
    showcase = [53, 73, 41, 82, 77, 25, 5, 72, 93, 0, 86, 47][:n_display]
    class_images = {}
    for img, label in dataset_resized:
        if label in showcase and label not in class_images:
            class_images[label] = img
        if len(class_images) == n_display:
            break

    fig, axes = plt.subplots(n_display // 3, 3, figsize=(16, 4 * (n_display // 3)))
    fig.suptitle("Oxford 102 Flowers — Images with Text Descriptions",
                 fontsize=14, fontweight='bold')

    for i, cls in enumerate(showcase):
        ax = axes[i // 3][i % 3]
        if cls in class_images:
            img = class_images[cls]
            img_display = (img * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
            ax.imshow(img_display)
        name, desc = FLOWER_DESCRIPTIONS[cls]
        wrapped = "\n".join([desc[j:j+45] for j in range(0, len(desc), 45)])
        ax.set_title(f"{name.upper()}\n{wrapped}",
                     fontsize=7, loc='left', pad=4)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("outputs/task4_images_with_descriptions.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: outputs/task4_images_with_descriptions.png")


def plot_aspect_ratio_distribution(stats: dict):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(stats['aspect_ratios'], bins=30, color='mediumpurple',
            edgecolor='white', linewidth=0.5)
    ax.axvline(1.0, color='red', linestyle='--', label="Square (1:1)")
    ax.axvline(np.mean(stats['aspect_ratios']), color='orange',
               linestyle='--', label=f"Mean: {np.mean(stats['aspect_ratios']):.2f}")
    ax.set_title("Image Aspect Ratio Distribution", fontsize=12, fontweight='bold')
    ax.set_xlabel("Aspect Ratio (Width / Height)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/task4_aspect_ratios.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: outputs/task4_aspect_ratios.png")

def run_task4():
    print("=" * 60)
    print("  TASK 4: Dataset Exploration & Analysis")
    print("  Oxford 102 Flowers")
    print("=" * 60)

    os.makedirs("outputs", exist_ok=True)
    print("\n[1/5] Loading dataset...")
    dataset_raw     = load_dataset_for_analysis()
    dataset_resized = load_dataset_resized()
    print(f"Raw dataset    : {len(dataset_raw)} images")
    print(f"Resized dataset: {len(dataset_resized)} images")
    print("\n[2/5] Computing image")
    stats = compute_dataset_statistics(dataset_raw, n_samples=500)
    print("\n[3/5] Analyzing text descriptions")
    word_counts, char_counts, word_freq = analyze_descriptions()
    print("\n[4/5] Plotting statistics dashboard...")
    plot_dataset_statistics(stats, word_counts, char_counts, word_freq)
    plot_class_distribution(stats)
    plot_aspect_ratio_distribution(stats)
    print("\n[5/5] Displaying images with text descriptions...")
    display_images_with_descriptions(dataset_resized)
    print("  Task 4 Complete! Output files:")
    print("  - task4_statistics_dashboard.png")
    print("  - task4_class_distribution.png")
    print("  - task4_aspect_ratios.png")
    print("  - task4_images_with_descriptions.png")

run_task4()