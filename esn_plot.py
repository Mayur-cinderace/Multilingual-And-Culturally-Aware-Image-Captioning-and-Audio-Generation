# esn_hyperparam_plots.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from cultural_context import EnhancedESNCulturalController
from sklearn.manifold import TSNE  # optional, needs pip install scikit-learn
import random

def generate_training_samples(total=1000):
    samples = []

    food_bases = [
        "Butter chicken", "Paneer tikka masala", "Dal makhani", "Vegetable biryani",
        "Aloo paratha", "Masala dosa", "Rogan josh", "Chole bhature", "Palak paneer",
        "Malai kofta", "Baingan bharta", "Fish curry", "Pork vindaloo", "Chettinad chicken"
    ]
    festival_bases = [
        "Diwali diyas", "Holi colors", "Rangoli design", "Ganesh idol", "Durga Puja pandal",
        "Navratri garba", "Eid biryani", "Karwa Chauth thali", "Onam sadhya", "Pongal kolam"
    ]
    daily_bases = [
        "Morning chai", "Dal rice", "Roti sabzi", "Idli sambar", "Poha breakfast",
        "Curd rice", "Khichdi", "Tiffin lunch", "Evening samosa", "Home-cooked meal"
    ]
    generic_bases = [
        "Mountain lake", "City skyline", "Person reading", "Highway traffic", "Abstract painting",
        "Laptop on desk", "Rainy window", "Bird in flight", "Sunset reflection", "Snowy peak",
        "Empty beach", "Forest path", "Street lamp", "Bicycle parked", "Wooden chair"
    ]

    targets = [
        ("food_traditional", 250, food_bases, ["with naan", "in rich gravy", "served hot", "garnished with coriander", "slow-cooked", "spicy and aromatic"]),
        ("festival_context", 250, festival_bases, ["celebration", "with diyas", "colorful rangoli", "family gathering", "traditional attire", "puja ceremony"]),
        ("daily_life", 250, daily_bases, ["at home", "for breakfast", "family meal", "quick lunch", "evening snack", "homemade"]),
        ("generic", 250, generic_bases, ["at sunset", "in city", "in park", "on highway", "on wall", "at night"])
    ]

    for cls, count, bases, suffixes in targets:
        for _ in range(count):
            base = random.choice(bases)
            suffix = random.choice(suffixes)
            caption = f"{base} {suffix}.".strip()
            samples.append((caption, cls))

    random.shuffle(samples)
    return samples[:total]

# Usage:
training_samples = generate_training_samples(1000)
# ────────────────────────────────────────────────
# Test / Validation data (expand this!)
# ───────────────────────────────────────────────

def generate_large_test_set(size=1000):
    bases = {
        "food_traditional": [
            "Butter chicken", "Paneer tikka masala", "Dal makhani", "Vegetable biryani",
            "Aloo paratha", "Masala dosa", "Rogan josh", "Chole bhature", "Palak paneer",
            "Malai kofta", "Baingan bharta", "Fish curry", "Vindaloo", "Chettinad chicken"
        ],
        "festival_context": [
            "Diwali", "Holi", "Ganesh Chaturthi", "Durga Puja", "Navratri", "Eid",
            "Karwa Chauth", "Onam", "Pongal", "Baisakhi", "Bihu", "Rangoli", "Diya",
            "Prasad", "Modak", "Garba", "Dandiya", "Sheer khurma"
        ],
        "daily_life": [
            "Morning chai", "Filter coffee", "Roti sabzi", "Dal rice", "Poha breakfast",
            "Idli sambar", "Curd rice", "Khichdi", "Tiffin lunch", "Evening snack",
            "Family dinner", "Street vendor", "Home-cooked meal", "Grandmother cooking"
        ],
        "generic": [
            "Mountain landscape", "City skyline", "Person reading", "Car on highway",
            "Abstract painting", "Laptop on desk", "Rain on window", "Bird in tree",
            "Sunset reflection", "Snowy peak", "Open book", "Flower bouquet", "Train journey"
        ]
    }

    test_samples = []
    classes = ["food_traditional", "festival_context", "daily_life", "generic"]
    counts = [250, 250, 250, 250]  # balanced

    for cls, count in zip(classes, counts):
        for _ in range(count):
            base = random.choice(bases[cls])
            if cls == "food_traditional":
                suffix = random.choice(["with naan", "served hot", "garnished with coriander", "in rich gravy", "with rice", "slow-cooked", "spicy and aromatic"])
                caption = f"{base} {suffix}."
            elif cls == "festival_context":
                suffix = random.choice(["celebration", "with diyas", "colorful rangoli", "family gathering", "traditional attire", "puja ceremony", "offered as prasad"])
                caption = f"{base} {suffix}."
            elif cls == "daily_life":
                suffix = random.choice(["at home", "for breakfast", "family meal", "quick lunch", "evening snack", "street style", "homemade"])
                caption = f"{base} {suffix}."
            else:
                suffix = random.choice(["at sunset", "in city", "in park", "on highway", "on wall", "on desk", "at night", "in sky"])
                caption = f"{base} {suffix}."
            
            test_samples.append((caption, cls))

    random.shuffle(test_samples)
    return test_samples

test_samples = generate_large_test_set(1000)


def compute_acc(esn, samples):
    true = []
    pred = []
    for cap, mode in samples:
        p_mode, _ = esn.predict_mode(cap)
        true.append(esn.modes.index(mode))
        pred.append(esn.modes.index(p_mode))
    return np.mean(np.array(true) == np.array(pred))

# ────────────────────────────────────────────────
# 1. Spectral Radius Sweep
# ────────────────────────────────────────────────
spectral_radii = np.linspace(0.1, 1.5, 15)  # typical range 0.1–1.2
val_accs = []

for rho in spectral_radii:
    esn = EnhancedESNCulturalController(
        reservoir_size=200,
        spectral_radius=rho,
        leak_rate=0.35,
        input_scaling=0.3,
        sparsity=0.04
    )
    esn.train(training_samples)
    acc = compute_acc(esn, test_samples)
    val_accs.append(acc)
    print(f"Spectral radius {rho:.2f} → Val Acc: {acc:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(spectral_radii, val_accs, 'o-', color='teal')
plt.xlabel("Spectral Radius (ρ)")
plt.ylabel("Validation Accuracy")
plt.title("ESN: Spectral Radius vs Validation Accuracy")
plt.grid(True)
plt.savefig("esn_spectral_radius_vs_acc.png")
plt.show()

# ────────────────────────────────────────────────
# 2. Leak Rate (alpha) Sweep
# ────────────────────────────────────────────────
leak_rates = np.linspace(0.05, 1.0, 12)
val_accs_leak = []

for alpha in leak_rates:
    esn = EnhancedESNCulturalController(leak_rate=alpha)
    esn.train(training_samples)
    acc = compute_acc(esn, test_samples)
    val_accs_leak.append(acc)

plt.figure(figsize=(8, 5))
plt.plot(leak_rates, val_accs_leak, 'o-', color='purple')
plt.xlabel("Leak Rate (α)")
plt.ylabel("Validation Accuracy")
plt.title("ESN: Leak Rate vs Validation Accuracy")
plt.grid(True)
plt.savefig("esn_leak_rate_vs_acc.png")
plt.show()

# ────────────────────────────────────────────────
# 3. Reservoir Size Sweep (more expensive)
# ────────────────────────────────────────────────
sizes = [50, 100, 150, 200, 300, 400, 500]
val_accs_size = []

for sz in sizes:
    esn = EnhancedESNCulturalController(reservoir_size=sz)
    esn.train(training_samples)
    acc = compute_acc(esn, test_samples)
    val_accs_size.append(acc)

plt.figure(figsize=(8, 5))
plt.plot(sizes, val_accs_size, 'o-', color='darkgreen')
plt.xlabel("Reservoir Size (N)")
plt.ylabel("Validation Accuracy")
plt.title("ESN: Reservoir Size vs Validation Accuracy")
plt.grid(True)
plt.savefig("esn_size_vs_acc.png")
plt.show()

# ────────────────────────────────────────────────
# 4. Optional: Reservoir States Visualization (PCA or t-SNE)
# ────────────────────────────────────────────────
# Example: collect final states for test samples
esn = EnhancedESNCulturalController()  # best config
esn.train(training_samples)

states = []
labels = []
for cap, mode in test_samples[:200]:  # limit for speed
    # Run warm-up like in predict_mode
    esn.state.fill(0.0)
    x = esn.extract_features(cap)
    for _ in range(5):
        esn.state = (1 - esn.leak_rate) * esn.state + esn.leak_rate * np.tanh(esn.Win @ x + esn.W @ esn.state)
    states.append(esn.state.copy())
    labels.append(mode)

states = np.array(states)

# PCA
pca = PCA(n_components=2)
states_2d = pca.fit_transform(states)

plt.figure(figsize=(10, 7))
for mode in esn.modes:
    idx = [i for i, l in enumerate(labels) if l == mode]
    plt.scatter(states_2d[idx, 0], states_2d[idx, 1], label=mode, alpha=0.7)
plt.legend()
plt.title("ESN Reservoir States (PCA) - Class Separation")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.savefig("esn_reservoir_states_pca.png")
plt.show()

# Optional t-SNE (better separation but slower)
# tsne = TSNE(n_components=2, random_state=42)
# states_2d_tsne = tsne.fit_transform(states)
# ... same scatter plot ...