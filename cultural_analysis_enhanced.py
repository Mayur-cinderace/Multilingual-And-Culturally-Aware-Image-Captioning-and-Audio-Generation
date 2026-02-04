# cultural_analysis_enhanced.py
"""
Enhanced cultural analysis modules with rich interpretability.
Provides deep insights into cultural signals, model behavior, and trust metrics.
"""

import numpy as np
import json
from typing import Dict, List, Tuple
from datetime import datetime, timezone


class CulturalSignalAnalyzer:
    """
    Deep analysis of cultural signals with human-interpretable insights.
    """
    
    SIGNAL_DIMENSIONS = {
        "spice_level": {
            "name": "Culinary Spice Intensity",
            "low_desc": "Mild or neutral flavoring",
            "high_desc": "Rich, aromatic spice blends",
            "cultural_weight": 0.9
        },
        "ritual_score": {
            "name": "Ritual & Ceremony",
            "low_desc": "Secular or informal context",
            "high_desc": "Ceremonial or sacred practices",
            "cultural_weight": 1.0
        },
        "daily_score": {
            "name": "Everyday Life",
            "low_desc": "Special or unusual occasion",
            "high_desc": "Routine domestic activities",
            "cultural_weight": 0.7
        },
        "heritage_score": {
            "name": "Heritage & Tradition",
            "low_desc": "Modern or contemporary style",
            "high_desc": "Traditional or ancestral practices",
            "cultural_weight": 0.95
        },
        "has_people": {
            "name": "Social Presence",
            "low_desc": "Solitary or unpopulated",
            "high_desc": "Communal gathering or interaction",
            "cultural_weight": 0.8
        },
        "has_festival": {
            "name": "Festive Context",
            "low_desc": "Ordinary time",
            "high_desc": "Festival or celebration",
            "cultural_weight": 1.0
        },
        "has_thali": {
            "name": "Serving Tradition",
            "low_desc": "Individual serving",
            "high_desc": "Traditional multi-dish presentation",
            "cultural_weight": 0.85
        },
        "has_offering": {
            "name": "Devotional Elements",
            "low_desc": "Secular purpose",
            "high_desc": "Ritual offering or prasad",
            "cultural_weight": 0.95
        }
    }
    
    def analyze_signal_strength(self, caption: str, esn_model) -> Dict:
        """
        Comprehensive cultural signal analysis with interpretability.
        """
        features = esn_model.extract_features(caption)
        
        # Map features to dimensions
        dim_indices = {
            "spice_level": 1,
            "ritual_score": 2,
            "daily_score": 3,
            "heritage_score": 4,
            "has_people": 18,
            "has_festival": 9,
            "has_thali": 8,
            "has_offering": 16
        }
        
        analysis = {
            "dimensions": {},
            "dominant_signals": [],
            "cultural_intensity": 0.0,
            "interpretation": "",
            "confidence_factors": []
        }
        
        # Analyze each dimension
        total_weighted = 0.0
        total_weight = 0.0
        
        for dim_name, idx in dim_indices.items():
            if dim_name not in self.SIGNAL_DIMENSIONS:
                continue
                
            strength = float(features[idx])
            metadata = self.SIGNAL_DIMENSIONS[dim_name]
            weight = metadata["cultural_weight"]
            
            # Weighted contribution
            total_weighted += strength * weight
            total_weight += weight
            
            # Interpretation
            if strength >= 0.7:
                desc = metadata["high_desc"]
                level = "Strong"
            elif strength >= 0.4:
                desc = f"Moderate {metadata['name'].lower()}"
                level = "Moderate"
            else:
                desc = metadata["low_desc"]
                level = "Weak"
            
            analysis["dimensions"][metadata["name"]] = {
                "strength": round(strength, 3),
                "level": level,
                "description": desc,
                "weight": weight
            }
            
            # Track dominant signals
            if strength >= 0.65:
                analysis["dominant_signals"].append({
                    "name": metadata["name"],
                    "strength": round(strength, 3),
                    "description": desc
                })
        
        # Calculate overall cultural intensity
        analysis["cultural_intensity"] = round(total_weighted / total_weight, 3) if total_weight > 0 else 0.0
        
        # Generate interpretation
        analysis["interpretation"] = self._generate_interpretation(analysis)
        
        # Confidence factors
        analysis["confidence_factors"] = self._assess_confidence(analysis, features)
        
        return analysis
    
    def _generate_interpretation(self, analysis: Dict) -> str:
        """Generate human-readable interpretation."""
        intensity = analysis["cultural_intensity"]
        dominant = analysis["dominant_signals"]
        
        if intensity >= 0.75:
            base = "Strong cultural grounding detected."
        elif intensity >= 0.5:
            base = "Moderate cultural context identified."
        elif intensity >= 0.3:
            base = "Subtle cultural elements present."
        else:
            base = "Minimal cultural signaling."
        
        if len(dominant) >= 3:
            dims = ", ".join([d["name"] for d in dominant[:3]])
            return f"{base} Primary dimensions: {dims}."
        elif len(dominant) > 0:
            return f"{base} Notable: {dominant[0]['name']}."
        else:
            return base
    
    def _assess_confidence(self, analysis: Dict, features: np.ndarray) -> List[str]:
        """Assess confidence based on signal patterns."""
        factors = []
        
        intensity = analysis["cultural_intensity"]
        n_dominant = len(analysis["dominant_signals"])
        
        if n_dominant >= 3:
            factors.append("✓ Multiple converging signals")
        elif n_dominant == 0:
            factors.append("⚠ No dominant cultural markers")
        
        if intensity >= 0.7:
            factors.append("✓ High signal-to-noise ratio")
        elif intensity < 0.3:
            factors.append("⚠ Weak overall activation")
        
        # Check for ambiguity
        entropy = self._calculate_entropy(features[:20])
        if entropy > 2.5:
            factors.append("⚠ High signal dispersion (ambiguous)")
        else:
            factors.append("✓ Focused signal pattern")
        
        return factors
    
    def _calculate_entropy(self, features: np.ndarray) -> float:
        """Shannon entropy of feature distribution."""
        x = np.abs(features)
        x = x / (np.sum(x) + 1e-9)
        entropy = -np.sum(x * np.log(x + 1e-9))
        return float(entropy)


class ModelDisagreementAnalyzer:
    """
    Analyzes and explains ESN vs GRU disagreements.
    """
    
    def analyze(self, results: Dict, caption: str, esn_model, gru_model) -> Dict:
        """
        Deep analysis of model disagreement with explanations.
        """
        if not results or "esn" not in results or "gru" not in results:
            return {"status": "No predictions available"}
        
        esn = results["esn"]
        gru = results["gru"]
        
        disagreement = esn["mode"] != gru["mode"]
        conf_gap = abs(esn["confidence"] - gru["confidence"])
        
        analysis = {
            "disagreement": disagreement,
            "severity": self._assess_severity(disagreement, conf_gap),
            "esn_prediction": {
                "mode": esn["mode"],
                "confidence": round(esn["confidence"], 3),
                "reasoning": self._explain_esn(esn, caption, esn_model)
            },
            "gru_prediction": {
                "mode": gru["mode"],
                "confidence": round(gru["confidence"], 3),
                "reasoning": self._explain_gru(gru, caption)
            },
            "confidence_gap": round(conf_gap, 3),
            "interpretation": "",
            "recommendation": ""
        }
        
        # Generate interpretation
        analysis["interpretation"] = self._generate_disagreement_interpretation(analysis)
        
        # Recommendation
        analysis["recommendation"] = self._generate_recommendation(analysis)
        
        return analysis
    
    def _assess_severity(self, disagreement: bool, conf_gap: float) -> str:
        """Assess disagreement severity."""
        if not disagreement:
            return "None"
        elif conf_gap > 0.3:
            return "High"
        elif conf_gap > 0.15:
            return "Moderate"
        else:
            return "Low"
    
    def _explain_esn(self, esn: Dict, caption: str, esn_model) -> str:
        """Explain ESN's reasoning."""
        features = esn_model.extract_features(caption)
        
        # Get top 3 features
        feature_names = [
            "length", "spice", "ritual", "daily", "heritage",
            "color", "texture", "serving", "thali", "festival"
        ]
        
        top_idx = np.argsort(features[:10])[::-1][:3]
        top_features = [feature_names[i] for i in top_idx if features[i] > 0.5]
        
        if top_features:
            return f"Activated by: {', '.join(top_features)}"
        else:
            return "Weak feature activation"
    
    def _explain_gru(self, gru: Dict, caption: str) -> str:
        """Explain GRU's reasoning."""
        conf = gru["confidence"]
        
        if conf > 0.7:
            return "High confidence from sequential pattern recognition"
        elif conf > 0.5:
            return "Moderate confidence from learned patterns"
        else:
            return "Low confidence, uncertain classification"
    
    def _generate_disagreement_interpretation(self, analysis: Dict) -> str:
        """Generate human-readable interpretation."""
        if not analysis["disagreement"]:
            return "Both models converge on the same classification, indicating stable cultural interpretation."
        
        severity = analysis["severity"]
        esn_mode = analysis["esn_prediction"]["mode"]
        gru_mode = analysis["gru_prediction"]["mode"]
        
        if severity == "High":
            return (
                f"Significant disagreement: ESN sees '{esn_mode}' while GRU sees '{gru_mode}'. "
                f"This suggests ambiguous or mixed cultural signals in the image."
            )
        elif severity == "Moderate":
            return (
                f"Models differ (ESN: {esn_mode}, GRU: {gru_mode}) but with moderate confidence gap. "
                f"The image likely contains subtle cultural markers open to interpretation."
            )
        else:
            return (
                f"Minor disagreement between models. Both are uncertain, "
                f"suggesting the image has weak cultural context."
            )
    
    def _generate_recommendation(self, analysis: Dict) -> str:
        """Generate actionable recommendation."""
        if not analysis["disagreement"]:
            return "✓ Trust both predictions - they agree."
        
        esn_conf = analysis["esn_prediction"]["confidence"]
        gru_conf = analysis["gru_prediction"]["confidence"]
        
        if esn_conf > gru_conf + 0.2:
            return f"→ Prefer ESN prediction ('{analysis['esn_prediction']['mode']}') - stronger immediate signal detection."
        elif gru_conf > esn_conf + 0.2:
            return f"→ Prefer GRU prediction ('{analysis['gru_prediction']['mode']}') - better learned pattern matching."
        else:
            return "→ Treat as 'generic' due to model uncertainty."


class CulturalSurpriseDetector:
    """
    Detects violations of cultural expectations.
    """
    
    def detect(self, caption: str, prediction: Dict, esn_model) -> Dict:
        """
        Detect and explain cultural surprise.
        """
        features = esn_model.extract_features(caption)
        entropy = self._calculate_entropy(features[:20])
        
        esn = prediction["esn"]
        gru = prediction["gru"]
        
        disagreement = esn["mode"] != gru["mode"]
        conf_gap = abs(esn["confidence"] - gru["confidence"])
        overall_conf = max(esn["confidence"], gru["confidence"])
        
        # Surprise criteria
        is_surprise = (
            disagreement and
            entropy > 2.2 and
            0.35 <= overall_conf <= 0.7 and
            conf_gap > 0.2
        )
        
        analysis = {
            "is_surprise": is_surprise,
            "surprise_level": self._assess_surprise_level(is_surprise, entropy, conf_gap),
            "entropy": round(entropy, 3),
            "esn_mode": esn["mode"],
            "gru_mode": gru["mode"],
            "confidence": round(overall_conf, 3),
            "explanation": "",
            "cultural_context": ""
        }
        
        # Generate explanation
        if is_surprise:
            analysis["explanation"] = (
                "This image violates learned cultural expectations. "
                "Multiple conflicting signals activate different interpretive frameworks, "
                "indicating a break from dominant patterns or hybrid cultural context."
            )
            analysis["cultural_context"] = self._explain_surprise_context(esn, gru)
        else:
            analysis["explanation"] = "No significant cultural expectation violation detected."
            analysis["cultural_context"] = "The image conforms to learned cultural patterns."
        
        return analysis
    
    def _calculate_entropy(self, features: np.ndarray) -> float:
        x = np.abs(features)
        x = x / (np.sum(x) + 1e-9)
        return float(-np.sum(x * np.log(x + 1e-9)))
    
    def _assess_surprise_level(self, is_surprise: bool, entropy: float, conf_gap: float) -> str:
        if not is_surprise:
            return "None"
        elif entropy > 2.8 and conf_gap > 0.3:
            return "High"
        elif entropy > 2.4:
            return "Moderate"
        else:
            return "Low"
    
    def _explain_surprise_context(self, esn: Dict, gru: Dict) -> str:
        """Explain the nature of the surprise."""
        esn_mode = esn["mode"]
        gru_mode = gru["mode"]
        
        transitions = {
            ("food_traditional", "generic"): "Traditional food elements in non-food context",
            ("festival_context", "daily_life"): "Festival markers in everyday setting",
            ("daily_life", "festival_context"): "Everyday activity during festive time",
            ("generic", "food_traditional"): "Unexpected traditional food elements"
        }
        
        key = (esn_mode, gru_mode)
        reverse_key = (gru_mode, esn_mode)
        
        if key in transitions:
            return transitions[key]
        elif reverse_key in transitions:
            return transitions[reverse_key]
        else:
            return f"Unexpected transition between {esn_mode} and {gru_mode}"


class CulturalSilenceDetector:
    """
    Detects meaningful absences in cultural signaling.
    """
    
    def detect(self, caption: str, esn_model) -> Dict:
        """
        Detect cultural silence (meaningful absence of cultural markers).
        """
        features = esn_model.extract_features(caption)
        
        # Key absence indicators
        silence_dimensions = {
            "people_absent": {
                "idx": 18,
                "threshold": 0.3,
                "meaning": "No social or communal presence"
            },
            "ritual_absent": {
                "idx": 2,
                "threshold": 0.25,
                "meaning": "No ceremonial or sacred elements"
            },
            "heritage_absent": {
                "idx": 4,
                "threshold": 0.3,
                "meaning": "No traditional or ancestral markers"
            },
            "offering_absent": {
                "idx": 16,
                "threshold": 0.2,
                "meaning": "No devotional or ritual objects"
            }
        }
        
        active_silences = []
        for name, config in silence_dimensions.items():
            if features[config["idx"]] < config["threshold"]:
                active_silences.append({
                    "dimension": name.replace("_absent", ""),
                    "strength": round(1.0 - features[config["idx"]], 3),
                    "meaning": config["meaning"]
                })
        
        is_silence = len(active_silences) >= 2
        
        analysis = {
            "is_silence": is_silence,
            "silence_level": self._assess_silence_level(len(active_silences)),
            "missing_signals": active_silences,
            "interpretation": "",
            "cultural_meaning": ""
        }
        
        if is_silence:
            analysis["interpretation"] = (
                f"Cultural silence detected across {len(active_silences)} dimensions. "
                "The image is marked by the absence of typical social, ritual, or heritage cues."
            )
            analysis["cultural_meaning"] = self._explain_silence_meaning(active_silences)
        else:
            analysis["interpretation"] = "No significant cultural silence detected."
            analysis["cultural_meaning"] = "The image contains typical cultural markers."
        
        return analysis
    
    def _assess_silence_level(self, n_silences: int) -> str:
        if n_silences >= 3:
            return "High"
        elif n_silences == 2:
            return "Moderate"
        else:
            return "Low"
    
    def _explain_silence_meaning(self, silences: List[Dict]) -> str:
        """Interpret what the silence means culturally."""
        if len(silences) >= 3:
            return (
                "Suggests isolation, private space, or transitional context. "
                "May indicate modern, secular, or individual-focused setting."
            )
        else:
            dims = ", ".join([s["dimension"] for s in silences])
            return f"Absence of {dims} suggests contextual shift or boundary crossing."


class CounterfactualFrameAnalyzer:
    """
    Analyzes how cultural interpretation shifts under different frames.
    """
    
    FRAME_DEFINITIONS = {
        "Post-pandemic urban": {
            "description": "Contemporary urban life after COVID-19",
            "cultural_shifts": [
                "Reduced communal gatherings",
                "Increased individual/nuclear family focus",
                "Hybrid traditional-modern practices"
            ]
        },
        "Rural 1980s": {
            "description": "Traditional rural life before globalization",
            "cultural_shifts": [
                "Strong heritage preservation",
                "Community-centered activities",
                "Less object/commodity focus"
            ]
        },
        "Minimalist modern": {
            "description": "Contemporary minimalist lifestyle",
            "cultural_shifts": [
                "Reduced social visibility",
                "Object-focused aesthetics",
                "Secularized practices"
            ]
        }
    }
    
    def analyze_shift(self, signal_vector: Dict, frame: str, modifiers: Dict) -> Dict:
        """
        Analyze how signals shift under counterfactual frame.
        """
        if not signal_vector:
            return {"error": "No signal vector available"}
        
        if frame not in self.FRAME_DEFINITIONS:
            return {"error": f"Unknown frame: {frame}"}
        
        # Apply modifiers
        adjusted = {}
        changes = []
        
        for signal, value in signal_vector.items():
            modifier = modifiers.get(signal, 1.0)
            new_value = round(min(1.0, value * modifier), 3)
            adjusted[signal] = new_value
            
            # Track significant changes
            if abs(new_value - value) > 0.15:
                direction = "increases" if new_value > value else "decreases"
                changes.append({
                    "signal": signal.replace("_", " ").title(),
                    "from": round(value, 3),
                    "to": new_value,
                    "direction": direction,
                    "magnitude": round(abs(new_value - value), 3)
                })
        
        # Identify dominant dimensions
        dominant = sorted(adjusted.items(), key=lambda x: -x[1])[:3]
        
        analysis = {
            "frame": frame,
            "frame_description": self.FRAME_DEFINITIONS[frame]["description"],
            "original_signals": signal_vector,
            "adjusted_signals": adjusted,
            "significant_changes": changes,
            "dominant_dimensions": [
                {"dimension": k.replace("_", " ").title(), "strength": v}
                for k, v in dominant
            ],
            "interpretation": self._generate_frame_interpretation(frame, changes),
            "cultural_implications": self.FRAME_DEFINITIONS[frame]["cultural_shifts"]
        }
        
        return analysis
    
    def _generate_frame_interpretation(self, frame: str, changes: List[Dict]) -> str:
        """Generate interpretation of frame shift."""
        if not changes:
            return f"Under the '{frame}' frame, the cultural interpretation remains stable."
        
        major_changes = [c for c in changes if c["magnitude"] > 0.25]
        
        if len(major_changes) >= 2:
            signals = ", ".join([c["signal"] for c in major_changes[:2]])
            return (
                f"Under the '{frame}' frame, significant cultural reinterpretation occurs. "
                f"Major shifts in: {signals}."
            )
        elif len(changes) > 0:
            signal = changes[0]["signal"]
            direction = changes[0]["direction"]
            return f"Under the '{frame}' frame, {signal} {direction}, subtly shifting interpretation."
        else:
            return f"The '{frame}' frame produces minimal reinterpretation."