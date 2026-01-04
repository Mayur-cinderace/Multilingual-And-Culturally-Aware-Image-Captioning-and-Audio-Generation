# esn_cultural_context.py (Improved 2025/2026 version)
import re
import numpy as np
from typing import List, Tuple

class CulturalContextController:
    """
    Echo State Network for CULTURAL MODE selection
    Improved version with better feature set, spectral radius control,
    leaky integration, ridge regression and better cultural separation
    """

    def __init__(self, reservoir_size=80, spectral_radius=0.95, leak_rate=0.4, input_scaling=0.25):
        self.reservoir_size = reservoir_size
        self.leak_rate = leak_rate
        self.spectral_radius = spectral_radius

        # Input scaling usually 0.1–0.4
        self.Win = np.random.randn(reservoir_size, 14) * input_scaling

        # Reservoir weights – sparse + spectral radius scaling
        W = np.random.randn(reservoir_size, reservoir_size)
        # sparsity ~ 3–10%
        mask = np.random.rand(reservoir_size, reservoir_size) < 0.06
        W = W * mask
        # Scale to desired spectral radius
        rho = np.max(np.abs(np.linalg.eigvals(W)))
        if rho > 0:
            W *= spectral_radius / rho
        self.W = W.astype(np.float32)

        self.W_out = None
        self.state = np.zeros(reservoir_size, dtype=np.float32)

        self.modes = [
            "food_traditional",
            "festival_context",
            "daily_life",
            "generic"
        ]

    # ────────────────────────────────────────────────
    # Much richer, yet still fast & interpretable features
    # ────────────────────────────────────────────────
    def extract_features(self, caption: str) -> np.ndarray:
        text = " " + caption.lower() + " "  # word boundary padding
        tokens = re.findall(r"\w+", text)
        n_tokens = len(tokens)

        # Basic normalization
        length_score = min(n_tokens / 25.0, 1.2)

        # ─── Food / Culinary domain ─────────────────────
        indian_spice_markers = sum(text.count(w) for w in [
            "masala", "tadka", "curry", "kadai", "tikka", "vindaloo", "vindalo", "korma",
            "rogan josh", "biryani", "biryani", "tandoori", "chaat", "paneer", "dal",
            "raita", "chutney", "sambar", "rasam", "ghee", "tadka", "jeera", "hing"
        ])
        spice_intensity = min(indian_spice_markers / 4.0, 1.5)

        serving_style = sum(1 for w in ["bowl", "thali", "plate", "tray", "served", "garnished", "topped", "sprinkled"]) if "thali" in text or "plate" in text else 0
        serving_score = min(serving_style / 3.5, 1.0)

        # ─── Ritual / Festival domain ───────────────────
        ritual_markers = sum(text.count(w) for w in [
            "festival", "diwali", "holi", "navratri", "onam", "pongal", "eid", "christmas",
            "celebration", "ritual", "offering", "puja", "aarti", "diya", "rangoli", "lamp",
            "prasad", "blessing", "sacred", "temple"
        ])
        ritual_score = min(ritual_markers / 3.0, 1.4)

        # ─── Everyday / Family domain ───────────────────
        daily_life_markers = sum(text.count(w) for w in [
            "home", "kitchen", "family", "mother", "grandma", "lunch", "dinner", "breakfast",
            "morning", "evening", "daily", "routine", "house", "homemade", "cooked"
        ])
        daily_score = min(daily_life_markers / 4.0, 1.2)

        # ─── Traditional / Heritage signals ─────────────
        heritage_markers = sum(text.count(w) for w in [
            "traditional", "authentic", "heritage", "ancestral", "generational", "classic",
            "age-old", "cultural", "regional", "village", "street", "local"
        ])
        heritage_score = min(heritage_markers / 3.5, 1.1)

        # Simple color + visual richness
        color_vivid = sum(text.count(w) for w in ["red", "saffron", "yellow", "golden", "green", "bright", "vibrant", "colorful"])
        color_score = min(color_vivid / 4.0, 1.0)

        # Very strong single indicators (binary + boosted)
        has_thali     = 1.5 if "thali" in text else 0.0
        has_festival  = 1.8 if any(w in text for w in ["festival", "diwali", "holi", "navratri"]) else 0.0
        has_home      = 1.0 if any(w in text for w in ["home", "homemade", "family"]) else 0.0

        features = np.array([
            length_score,
            spice_intensity,
            serving_score,
            ritual_score,
            daily_score,
            heritage_score,
            color_score,
            has_thali,
            has_festival,
            has_home,
            float("curry" in text or "biryani" in text or "paneer" in text),   # strong indian marker
            float(any(w in text for w in ["puja", "aarti", "prasad", "diya"])), # strong ritual
            1.0,   # bias
            0.15   # small noise / regularizer term
        ], dtype=np.float32)

        return features

    def predict_mode(self, caption: str) -> Tuple[str, float]:
        # Always reset state — we treat each caption independently
        self.state.fill(0.0)

        x = self.extract_features(caption)

        # Leaky integrator update
        input_drive = self.Win @ x
        reservoir_drive = self.W @ self.state
        update = input_drive + reservoir_drive
        new_state = np.tanh(update)
        self.state = (1 - self.leak_rate) * self.state + self.leak_rate * new_state

        if self.W_out is None:
            return "generic", 0.38

        logits = self.W_out @ self.state
        probs = np.exp(logits - np.max(logits))          # numerical stability
        probs /= np.sum(probs)

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

        return self.modes[idx], confidence

    def train(self, samples: List[Tuple[str, str]], ridge_alpha: float = 1e-4):
        """
        samples = list of (caption, target_mode)
        ridge_alpha: helps when number of samples is small
        """
        if not samples:
            return

        X_features = []
        states = []

        self.state.fill(0.0)

        for caption, mode_str in samples:
            x = self.extract_features(caption)
            # same update rule as inference
            input_drive = self.Win @ x
            reservoir_drive = self.W @ self.state
            update = input_drive + reservoir_drive
            new_state = np.tanh(update)
            self.state = (1 - self.leak_rate) * self.state + self.leak_rate * new_state

            X_features.append(x)
            states.append(self.state.copy())

        states = np.array(states)      # (n_samples, reservoir_size)
        targets = np.zeros((len(samples), len(self.modes)))

        for i, (_, mode_str) in enumerate(samples):
            if mode_str in self.modes:
                targets[i, self.modes.index(mode_str)] = 1.0
            else:
                # unknown mode → soft generic
                targets[i, self.modes.index("generic")] = 0.7

        # Ridge regression (more stable than pure pinv)
        I = np.eye(self.reservoir_size)
        self.W_out = np.linalg.solve(
            states.T @ states + ridge_alpha * I,
            states.T @ targets
        ).T   # shape: (n_modes, reservoir_size)

        print(f"Trained W_out shape: {self.W_out.shape}  (ridge α={ridge_alpha})")