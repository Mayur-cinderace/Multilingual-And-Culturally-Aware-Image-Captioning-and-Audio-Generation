# cultural_context.py – Enhanced Echo State Network + GRU for cultural mode classification
import re
import numpy as np
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.optim as optim


class EnhancedESNCulturalController:
    def __init__(self, reservoir_size=200, spectral_radius=0.92, leak_rate=0.35, input_scaling=0.3, sparsity=0.04):
        self.reservoir_size = reservoir_size
        self.leak_rate = leak_rate
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity

        # Expanded to 28-dimensional input for more robust features
        self.Win = np.random.randn(reservoir_size, 28) * input_scaling

        # Reservoir with controlled sparsity
        W = np.random.randn(reservoir_size, reservoir_size)
        mask = np.random.rand(reservoir_size, reservoir_size) < sparsity
        W = W * mask
        rho = np.max(np.abs(np.linalg.eigvals(W)))
        if rho > 0:
            W *= spectral_radius / rho
        self.W = W.astype(np.float32)

        self.W_out = None
        self.state = np.zeros(reservoir_size, dtype=np.float32)

        self.modes = ["food_traditional", "festival_context", "daily_life", "generic"]

    def extract_features(self, caption: str) -> np.ndarray:
        text = " " + caption.lower() + " "
        tokens = re.findall(r"\w+", text)
        n_tokens = len(tokens)

        # Expanded keyword sets for better coverage
        length_score   = min(n_tokens / 30.0, 1.3)  # Adjusted for longer captions
        spice_score    = min(sum(text.count(w) for w in ["masala","curry","tikka","biryani","paneer","dal","chutney","garam","spicy","tandoori","korma","vindaloo"]) / 6.0, 1.6)
        ritual_score   = min(sum(text.count(w) for w in ["festival","diwali","holi","puja","prasad","diya","rangoli","eid","navratri","ganesh","durga","karwa","chath"]) / 5.0, 1.5)
        daily_score    = min(sum(text.count(w) for w in ["home","family","kitchen","lunch","dinner","breakfast","meal","routine","everyday","household","gathering"]) / 5.0, 1.3)
        heritage_score = min(sum(text.count(w) for w in ["traditional","heritage","authentic","regional","ancient","cultural","grandmother","recipe","handed","down"]) / 4.0, 1.2)
        color_score    = min(sum(text.count(w) for w in ["yellow","golden","red","saffron","bright","vibrant","colorful","green","orange","turmeric"]) / 5.0, 1.1)
        texture_score  = min(sum(text.count(w) for w in ["crispy","creamy","soft","crunchy","fluffy","thick","thin","gravy","dry","fried","steamed","roasted"]) / 6.0, 1.0)
        serving_score  = min(sum(text.count(w) for w in ["plate","bowl","thali","served","garnished","topped","accompanied","side"]) / 4.0, 1.0)

        has_thali      = 1.6 if "thali" in text else 0.0
        has_festival   = 1.9 if any(w in text for w in ["festival","diwali","holi","eid"]) else 0.0
        has_sweet      = 1.3 if any(w in text for w in ["sweet","laddoo","barfi","jalebi","mithai","gulab","jamun","peda","rasgulla"]) else 0.0
        has_rice       = 1.1 if any(w in text for w in ["rice","biryani","pulao","khichdi","idli","dosa"]) else 0.0
        has_veg        = 0.9 if any(w in text for w in ["veg","paneer","vegetable","aloo","sabzi","gobi","baingan","bhindi","palak"]) else 0.0
        has_meat       = 0.9 if any(w in text for w in ["chicken","mutton","lamb","fish","meat","kebab","tandoori","rogan","josh"]) else 0.0
        has_dessert    = 1.0 if any(w in text for w in ["dessert","halwa","kheer","payasam","sheera"]) else 0.0
        has_beverage   = 0.8 if any(w in text for w in ["chai","tea","coffee","lassi","buttermilk","sharbat"]) else 0.0
        has_offering   = 1.4 if any(w in text for w in ["offering","prasad","bhog","temple","worship"]) else 0.0
        has_decoration = 1.2 if any(w in text for w in ["decoration","lights","flowers","rangoli","toran"]) else 0.0
        has_people     = 1.0 if any(w in text for w in ["people","family","friends","gathering","celebrating"]) else 0.0
        has_utensil    = 0.7 if any(w in text for w in ["utensil","kadai","tawa","pressure","cooker","brass","copper"]) else 0.0

        # Additional structural features
        has_adjective = sum(1 for w in tokens if w.endswith("ing") or w.endswith("ed")) / max(n_tokens, 1)  # Descriptive richness
        uniqueness    = len(set(tokens)) / max(n_tokens, 1)  # Vocabulary diversity
        sentiment     = sum(1 for w in ["delicious","aromatic","rich","flavorful","tasty","yummy"] if w in text) / 3.0  # Positive food sentiment

        return np.array([
            length_score, spice_score, ritual_score, daily_score, heritage_score,
            color_score, texture_score, serving_score, has_thali, has_festival,
            has_sweet, has_rice, has_veg, has_meat, has_dessert, has_beverage,
            has_offering, has_decoration, has_people, has_utensil,
            has_adjective, uniqueness, sentiment, 1.0, 0.15, 0.05, 0.1, 0.2  # biases + constants
        ], dtype=np.float32)

    def predict_mode(self, caption: str) -> Tuple[str, float]:
        if self.W_out is None:
            return "generic", 0.40

        self.state.fill(0.0)
        x = self.extract_features(caption)

        for t in range(5):  # Longer warm-up for better state evolution
            input_drive = self.Win @ x
            reservoir_drive = self.W @ self.state
            update = input_drive + reservoir_drive
            noise = np.random.randn(self.reservoir_size) * 0.005  # Small noise for robustness
            new_state = np.tanh(update + noise)
            self.state = (1 - self.leak_rate) * self.state + self.leak_rate * new_state

        logits = self.W_out @ self.state
        probs = np.exp(logits - np.max(logits)) / (np.sum(np.exp(logits - np.max(logits))) + 1e-12)
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

        return self.modes[idx], confidence

    def train(self, samples: List[Tuple[str, str]], ridge_alpha: float = 5e-5):
        states = []
        self.state.fill(0.0)

        for caption, _ in samples:
            x = self.extract_features(caption)
            for t in range(5):  # Matching predict warm-up
                input_drive = self.Win @ x
                reservoir_drive = self.W @ self.state
                update = input_drive + reservoir_drive
                noise = np.random.randn(self.reservoir_size) * 0.005
                new_state = np.tanh(update + noise)
                self.state = (1 - self.leak_rate) * self.state + self.leak_rate * new_state
            states.append(self.state.copy())

        states = np.array(states)
        targets = np.zeros((len(samples), len(self.modes)))
        for i, (_, mode_str) in enumerate(samples):
            idx = self.modes.index(mode_str) if mode_str in self.modes else 3
            targets[i, idx] = 1.0

        I = np.eye(self.reservoir_size)
        self.W_out = np.linalg.solve(states.T @ states + ridge_alpha * I, states.T @ targets).T
        print(f"Enhanced ESN trained – W_out shape: {self.W_out.shape}")


class EnhancedGRUCulturalController(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, output_size=4, num_layers=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.modes = ["food_traditional", "festival_context", "daily_life", "generic"]

        # Use PyTorch GRU for robustness
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def extract_features(self, caption: str) -> torch.Tensor:
        # Same expanded features as ESN
        text = " " + caption.lower() + " "
        tokens = re.findall(r"\w+", text)
        n_tokens = len(tokens)

        length_score   = min(n_tokens / 30.0, 1.3)
        spice_score    = min(sum(text.count(w) for w in ["masala","curry","tikka","biryani","paneer","dal","chutney","garam","spicy","tandoori","korma","vindaloo"]) / 6.0, 1.6)
        ritual_score   = min(sum(text.count(w) for w in ["festival","diwali","holi","puja","prasad","diya","rangoli","eid","navratri","ganesh","durga","karwa","chath"]) / 5.0, 1.5)
        daily_score    = min(sum(text.count(w) for w in ["home","family","kitchen","lunch","dinner","breakfast","meal","routine","everyday","household","gathering"]) / 5.0, 1.3)
        heritage_score = min(sum(text.count(w) for w in ["traditional","heritage","authentic","regional","ancient","cultural","grandmother","recipe","handed","down"]) / 4.0, 1.2)
        color_score    = min(sum(text.count(w) for w in ["yellow","golden","red","saffron","bright","vibrant","colorful","green","orange","turmeric"]) / 5.0, 1.1)
        texture_score  = min(sum(text.count(w) for w in ["crispy","creamy","soft","crunchy","fluffy","thick","thin","gravy","dry","fried","steamed","roasted"]) / 6.0, 1.0)
        serving_score  = min(sum(text.count(w) for w in ["plate","bowl","thali","served","garnished","topped","accompanied","side"]) / 4.0, 1.0)

        has_thali      = 1.6 if "thali" in text else 0.0
        has_festival   = 1.9 if any(w in text for w in ["festival","diwali","holi","eid"]) else 0.0
        has_sweet      = 1.3 if any(w in text for w in ["sweet","laddoo","barfi","jalebi","mithai","gulab","jamun","peda","rasgulla"]) else 0.0
        has_rice       = 1.1 if any(w in text for w in ["rice","biryani","pulao","khichdi","idli","dosa"]) else 0.0
        has_veg        = 0.9 if any(w in text for w in ["veg","paneer","vegetable","aloo","sabzi","gobi","baingan","bhindi","palak"]) else 0.0
        has_meat       = 0.9 if any(w in text for w in ["chicken","mutton","lamb","fish","meat","kebab","tandoori","rogan","josh"]) else 0.0
        has_dessert    = 1.0 if any(w in text for w in ["dessert","halwa","kheer","payasam","sheera"]) else 0.0
        has_beverage   = 0.8 if any(w in text for w in ["chai","tea","coffee","lassi","buttermilk","sharbat"]) else 0.0
        has_offering   = 1.4 if any(w in text for w in ["offering","prasad","bhog","temple","worship"]) else 0.0
        has_decoration = 1.2 if any(w in text for w in ["decoration","lights","flowers","rangoli","toran"]) else 0.0
        has_people     = 1.0 if any(w in text for w in ["people","family","friends","gathering","celebrating"]) else 0.0
        has_utensil    = 0.7 if any(w in text for w in ["utensil","kadai","tawa","pressure","cooker","brass","copper"]) else 0.0

        has_adjective = sum(1 for w in tokens if w.endswith("ing") or w.endswith("ed")) / max(n_tokens, 1)
        uniqueness    = len(set(tokens)) / max(n_tokens, 1)
        sentiment     = sum(1 for w in ["delicious","aromatic","rich","flavorful","tasty","yummy"] if w in text) / 3.0

        features = [
            length_score, spice_score, ritual_score, daily_score, heritage_score,
            color_score, texture_score, serving_score, has_thali, has_festival,
            has_sweet, has_rice, has_veg, has_meat, has_dessert, has_beverage,
            has_offering, has_decoration, has_people, has_utensil,
            has_adjective, uniqueness, sentiment, 1.0, 0.15, 0.05, 0.1, 0.2
        ]

        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        logits = self.fc(output[:, -1, :])
        return logits

    def predict_mode(self, caption: str) -> Tuple[str, float]:
        x = self.extract_features(caption)
        self.eval()
        with torch.no_grad():
            logits = self.forward(x).squeeze()
            probs = torch.softmax(logits, dim=0).cpu().numpy()
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        return self.modes[idx], confidence

    def train_model(self, samples: List[Tuple[str, str]], epochs=25, lr=0.005):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        targets = torch.tensor([self.modes.index(mode) if mode in self.modes else 3 for _, mode in samples], dtype=torch.long)

        for epoch in range(epochs):
            total_loss = 0.0
            features = torch.cat([self.extract_features(caption) for caption, _ in samples], dim=0)  # (N, 1, input_size)

            self.train()
            optimizer.zero_grad()
            logits = self.forward(features).squeeze(1)  # (N, output_size)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(samples)
        print(f"GRU epoch {epoch+1}/{epochs} – avg loss: {avg_loss:.4f}")

        print("Enhanced GRU training completed.")


class CulturalContextManager:
    def __init__(self):
        self.esn = EnhancedESNCulturalController()
        self.gru = EnhancedGRUCulturalController()

    def train(self, samples):
        print("Training Enhanced ESN...")
        self.esn.train(samples)
        print("\nTraining Enhanced GRU...")
        self.gru.train_model(samples, epochs=20, lr=0.004)

    def predict(self, caption: str) -> Dict:
        esn_mode, esn_conf = self.esn.predict_mode(caption)
        gru_mode, gru_conf = self.gru.predict_mode(caption)

        return {
            "esn": {"mode": esn_mode, "confidence": esn_conf},
            "gru": {"mode": gru_mode, "confidence": gru_conf},
            "combined_mode": esn_mode if esn_conf > gru_conf else gru_mode
        }