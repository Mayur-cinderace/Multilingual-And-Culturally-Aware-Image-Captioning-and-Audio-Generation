# train_and_plot_models.py
# Standalone script to train and plot epoch-related metrics for GRU (and final for ESN)
# Run this in your environment with cultural_context available

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from cultural_context import EnhancedESNCulturalController, EnhancedGRUCulturalController
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

# Modes
modes = ["food_traditional", "festival_context", "daily_life", "generic"]
mode_to_idx = {m: i for i, m in enumerate(modes)}

# Function to compute accuracy
def compute_acc(model, samples):
    true = [mode_to_idx[mode] for _, mode in samples]
    pred = []
    for cap, _ in samples:
        pred_mode, _ = model.predict_mode(cap)
        pred.append(mode_to_idx[pred_mode])
    return accuracy_score(true, pred)

# Train ESN (one-shot, no epochs)
esn = EnhancedESNCulturalController()
esn.train(training_samples)

# ESN final metrics
esn_train_acc = compute_acc(esn, training_samples)
esn_val_acc = compute_acc(esn, test_samples)
print(f"ESN (one-shot): Train Acc = {esn_train_acc:.4f}, Val Acc = {esn_val_acc:.4f}")

# No plot for ESN since no epochs

# Train GRU with per-epoch metrics
gru = EnhancedGRUCulturalController()
optimizer = optim.Adam(gru.parameters(), lr=0.004)
criterion = nn.CrossEntropyLoss()

targets = torch.tensor([gru.modes.index(mode) if mode in gru.modes else 3 for _, mode in training_samples], dtype=torch.long)
gru_loss_history = []
gru_train_acc_history = []
gru_val_acc_history = []

features = torch.cat([gru.extract_features(caption) for caption, _ in training_samples], dim=0)  # (N, 1, input_size)

for epoch in range(20):
    gru.train()
    optimizer.zero_grad()
    logits = gru.forward(features).squeeze(1)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    
    gru_loss_history.append(loss.item())
    
    # Train acc
    pred = logits.argmax(1).cpu().numpy()
    train_acc = accuracy_score(targets.cpu().numpy(), pred)
    gru_train_acc_history.append(train_acc)
    
    # Val acc
    gru.eval()
    val_acc = compute_acc(gru, test_samples)
    gru_val_acc_history.append(val_acc)
    
    print(f"GRU Epoch {epoch+1}/20 - Loss: {loss.item():.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

# Plot for GRU: Loss vs Epoch
plt.figure(figsize=(10, 5))
plt.plot(range(1, 21), gru_loss_history, label="Training Loss", color="red")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GRU Training Loss vs Epoch")
plt.legend()
plt.grid(True)
plt.savefig("gru_loss_vs_epoch.png")
plt.show()

# Plot for GRU: Accuracy vs Epoch
plt.figure(figsize=(10, 5))
plt.plot(range(1, 21), gru_train_acc_history, label="Train Accuracy", color="blue")
plt.plot(range(1, 21), gru_val_acc_history, label="Validation Accuracy", color="green")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("GRU Accuracy vs Epoch")
plt.legend()
plt.grid(True)
plt.savefig("gru_accuracy_vs_epoch.png")
plt.show()

# Optional: Combined plot
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(range(1, 21), gru_loss_history, label="Loss", color="red")
ax1.set_ylabel("Loss")
ax1.set_xlabel("Epoch")

ax2 = ax1.twinx()
ax2.plot(range(1, 21), gru_train_acc_history, label="Train Acc", color="blue")
ax2.plot(range(1, 21), gru_val_acc_history, label="Val Acc", color="green")
ax2.set_ylabel("Accuracy")

fig.suptitle("GRU Training Metrics vs Epoch")
fig.legend(loc="upper right")
plt.grid(True)
plt.savefig("gru_combined_metrics.png")
plt.show()

print("Plots saved: gru_loss_vs_epoch.png, gru_accuracy_vs_epoch.png, gru_combined_metrics.png")
print("For ESN (no epochs): Use hyperparameter tuning plots if needed (e.g., spectral_radius vs acc)")