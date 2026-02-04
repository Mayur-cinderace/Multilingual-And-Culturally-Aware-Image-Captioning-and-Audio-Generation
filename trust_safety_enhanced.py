# trust_safety_enhanced.py
"""
Enhanced trust & safety metrics with cultural grounding.
Provides comprehensive evaluation of model behavior and outputs.
"""

import numpy as np
from typing import Dict, List, Tuple


class CulturalTrustAnalyzer:
    """
    Multi-dimensional trust analysis for cultural AI systems.
    """
    
    def analyze_comprehensive(self, prediction: Dict, caption: str, 
                            signal_vector: Dict, esn_model) -> Dict:
        """
        Comprehensive trust and safety analysis.
        """
        
        if not prediction:
            return {
                "status": "error",
                "message": "No prediction available"
            }
        
        mode = prediction.get("combined_mode", "generic")
        esn_conf = prediction.get("esn", {}).get("confidence", 0.0)
        gru_conf = prediction.get("gru", {}).get("confidence", 0.0)
        
        analysis = {
            # Core metrics
            "hallucination_risk": self._assess_hallucination_risk(esn_conf, gru_conf, mode),
            "vision_text_consistency": self._assess_vision_consistency(caption, signal_vector, mode),
            "cultural_confidence": self._assess_cultural_confidence(esn_conf, gru_conf, signal_vector),
            
            # New cultural metrics
            "cultural_authenticity": self._assess_authenticity(signal_vector, mode),
            "stereotype_risk": self._assess_stereotype_risk(caption, mode, esn_model),
            "representation_quality": self._assess_representation(signal_vector, mode),
            
            # Composite scores
            "overall_trust_score": 0.0,
            "safety_level": "",
            
            # Interpretability
            "trust_factors": [],
            "risk_factors": [],
            "recommendations": []
        }
        
        # Calculate composite
        analysis["overall_trust_score"] = self._calculate_overall_trust(analysis)
        analysis["safety_level"] = self._determine_safety_level(analysis["overall_trust_score"])
        
        # Generate factors
        analysis["trust_factors"] = self._identify_trust_factors(analysis)
        analysis["risk_factors"] = self._identify_risk_factors(analysis)
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _assess_hallucination_risk(self, esn_conf: float, gru_conf: float, mode: str) -> Dict:
        """
        Assess risk of model hallucinating cultural context.
        """
        max_conf = max(esn_conf, gru_conf)
        min_conf = min(esn_conf, gru_conf)
        conf_gap = max_conf - min_conf
        
        # High disagreement + low confidence = high hallucination risk
        if conf_gap > 0.3 and max_conf < 0.6:
            risk = 0.75
            level = "High"
            explanation = "Models disagree with low confidence - may be hallucinating patterns"
        elif conf_gap > 0.2:
            risk = 0.5
            level = "Moderate"
            explanation = "Some model uncertainty - verify cultural claims"
        elif max_conf < 0.5:
            risk = 0.6
            level = "Moderate"
            explanation = "Low overall confidence - cultural interpretation uncertain"
        else:
            risk = 1.0 - max_conf
            level = "Low" if risk < 0.3 else "Moderate"
            explanation = "Models relatively confident and aligned"
        
        return {
            "score": round(risk, 3),
            "level": level,
            "explanation": explanation,
            "confidence_gap": round(conf_gap, 3)
        }
    
    def _assess_vision_consistency(self, caption: str, signal_vector: Dict, mode: str) -> Dict:
        """
        Assess alignment between vision-grounded caption and cultural classification.
        """
        # Check if cultural mode aligns with caption content
        caption_lower = caption.lower()
        
        alignment_indicators = {
            "food_traditional": ["curry", "rice", "dish", "meal", "spice", "food", "traditional"],
            "festival_context": ["festival", "celebration", "diwali", "holi", "decoration"],
            "daily_life": ["home", "family", "everyday", "routine", "simple"],
            "generic": []
        }
        
        indicators = alignment_indicators.get(mode, [])
        
        if mode == "generic":
            score = 0.65
            level = "Moderate"
            explanation = "Generic classification - limited cultural context to verify"
        elif not indicators:
            score = 0.5
            level = "Low"
            explanation = "Unknown mode - cannot assess alignment"
        else:
            matches = sum(1 for ind in indicators if ind in caption_lower)
            score = min(0.95, 0.5 + (matches * 0.15))
            
            if score >= 0.8:
                level = "High"
                explanation = f"Strong alignment - caption mentions {matches} mode-relevant terms"
            elif score >= 0.6:
                level = "Moderate"
                explanation = f"Moderate alignment - some ({matches}) mode indicators present"
            else:
                level = "Low"
                explanation = "Weak alignment - caption doesn't strongly support classification"
        
        return {
            "score": round(score, 3),
            "level": level,
            "explanation": explanation
        }
    
    def _assess_cultural_confidence(self, esn_conf: float, gru_conf: float, 
                                   signal_vector: Dict) -> Dict:
        """
        Assess confidence in cultural interpretation.
        """
        max_conf = max(esn_conf, gru_conf)
        
        # Check signal strength
        if signal_vector:
            strong_signals = sum(1 for v in signal_vector.values() if v >= 0.7)
            signal_boost = min(0.2, strong_signals * 0.05)
        else:
            signal_boost = 0.0
        
        cultural_conf = min(1.0, max_conf + signal_boost)
        
        if cultural_conf >= 0.75:
            level = "High"
            explanation = "Strong cultural signals with model confidence"
        elif cultural_conf >= 0.55:
            level = "Moderate"
            explanation = "Moderate cultural grounding"
        else:
            level = "Low"
            explanation = "Weak cultural signals - interpretation uncertain"
        
        return {
            "score": round(cultural_conf, 3),
            "level": level,
            "explanation": explanation
        }
    
    def _assess_authenticity(self, signal_vector: Dict, mode: str) -> Dict:
        """
        Assess authenticity of cultural representation.
        """
        if not signal_vector or mode == "generic":
            return {
                "score": 0.5,
                "level": "N/A",
                "explanation": "Generic content - authenticity not applicable"
            }
        
        # Authenticity based on signal coherence
        values = list(signal_vector.values())
        
        # Check for coherent pattern (not scattered weak signals)
        strong_signals = sum(1 for v in values if v >= 0.7)
        weak_signals = sum(1 for v in values if v < 0.4)
        
        if strong_signals >= 2:
            score = 0.85
            level = "High"
            explanation = "Multiple strong cultural signals indicate authentic representation"
        elif strong_signals >= 1 and weak_signals <= 3:
            score = 0.65
            level = "Moderate"
            explanation = "Some authentic elements present"
        else:
            score = 0.4
            level = "Low"
            explanation = "Scattered or weak signals - authenticity uncertain"
        
        return {
            "score": round(score, 3),
            "level": level,
            "explanation": explanation
        }
    
    def _assess_stereotype_risk(self, caption: str, mode: str, esn_model) -> Dict:
        """
        Assess risk of stereotypical or reductive cultural representation.
        """
        features = esn_model.extract_features(caption)
        
        # Check for over-reliance on single dimension
        top_3_features = sorted(features[:20], reverse=True)[:3]
        
        if len(top_3_features) > 0:
            dominance = top_3_features[0] / (sum(top_3_features) + 1e-9)
        else:
            dominance = 0.0
        
        # High dominance = potential stereotyping
        if dominance > 0.7:
            risk = 0.7
            level = "Moderate"
            explanation = "Single cultural dimension dominates - may be reductive"
        elif dominance > 0.5:
            risk = 0.4
            level = "Low"
            explanation = "Some dimensional concentration - monitor for stereotyping"
        else:
            risk = 0.2
            level = "Low"
            explanation = "Balanced multi-dimensional representation"
        
        return {
            "score": round(risk, 3),
            "level": level,
            "explanation": explanation,
            "dominance": round(dominance, 3)
        }
    
    def _assess_representation(self, signal_vector: Dict, mode: str) -> Dict:
        """
        Assess quality of cultural representation.
        """
        if not signal_vector or mode == "generic":
            return {
                "score": 0.5,
                "level": "N/A",
                "explanation": "Generic content"
            }
        
        # Quality = diversity + strength + coherence
        values = list(signal_vector.values())
        
        diversity = len([v for v in values if v >= 0.5]) / max(len(values), 1)
        avg_strength = sum(values) / len(values) if values else 0
        
        quality = (diversity * 0.6 + avg_strength * 0.4)
        
        if quality >= 0.65:
            level = "High"
            explanation = "Rich, multi-dimensional cultural representation"
        elif quality >= 0.45:
            level = "Moderate"
            explanation = "Adequate cultural depth"
        else:
            level = "Low"
            explanation = "Thin cultural representation"
        
        return {
            "score": round(quality, 3),
            "level": level,
            "explanation": explanation
        }
    
    def _calculate_overall_trust(self, analysis: Dict) -> float:
        """Calculate composite trust score."""
        weights = {
            "hallucination_risk": -0.25,  # Negative impact
            "vision_text_consistency": 0.25,
            "cultural_confidence": 0.20,
            "cultural_authenticity": 0.15,
            "stereotype_risk": -0.10,  # Negative impact
            "representation_quality": 0.15
        }
        
        score = 0.5  # Baseline
        
        for metric, weight in weights.items():
            if metric in analysis and "score" in analysis[metric]:
                score += analysis[metric]["score"] * weight
        
        return round(max(0.0, min(1.0, score)), 3)
    
    def _determine_safety_level(self, trust_score: float) -> str:
        """Determine safety level from trust score."""
        if trust_score >= 0.75:
            return "✓ High Trust"
        elif trust_score >= 0.55:
            return "⚠ Moderate Trust"
        else:
            return "⚠ Low Trust - Review Required"
    
    def _identify_trust_factors(self, analysis: Dict) -> List[str]:
        """Identify positive trust factors."""
        factors = []
        
        if analysis["vision_text_consistency"]["score"] >= 0.7:
            factors.append("✓ Strong vision-text alignment")
        
        if analysis["cultural_confidence"]["score"] >= 0.7:
            factors.append("✓ High cultural confidence")
        
        if analysis["cultural_authenticity"]["score"] >= 0.7:
            factors.append("✓ Authentic representation")
        
        if analysis["stereotype_risk"]["score"] < 0.4:
            factors.append("✓ Low stereotype risk")
        
        if analysis["hallucination_risk"]["score"] < 0.3:
            factors.append("✓ Low hallucination risk")
        
        return factors if factors else ["⚠ No strong trust factors identified"]
    
    def _identify_risk_factors(self, analysis: Dict) -> List[str]:
        """Identify risk factors."""
        risks = []
        
        if analysis["hallucination_risk"]["score"] > 0.6:
            risks.append("⚠ High hallucination risk")
        
        if analysis["vision_text_consistency"]["score"] < 0.5:
            risks.append("⚠ Weak vision-text consistency")
        
        if analysis["stereotype_risk"]["score"] > 0.6:
            risks.append("⚠ Potential stereotyping")
        
        if analysis["representation_quality"]["score"] < 0.4:
            risks.append("⚠ Low representation quality")
        
        return risks if risks else ["✓ No significant risks detected"]
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recs = []
        
        trust_score = analysis["overall_trust_score"]
        
        if trust_score < 0.6:
            recs.append("→ Review cultural interpretation manually")
        
        if analysis["hallucination_risk"]["score"] > 0.5:
            recs.append("→ Verify cultural claims against source material")
        
        if analysis["stereotype_risk"]["score"] > 0.5:
            recs.append("→ Check for reductive or stereotypical framing")
        
        if analysis["vision_text_consistency"]["score"] < 0.6:
            recs.append("→ Ensure caption matches visual content")
        
        if not recs:
            recs.append("✓ Output appears trustworthy - proceed with confidence")
        
        return recs


class ModelBrainAnalyzer:
    """
    Deep inspection of model internal states and decision-making.
    """
    
    def analyze_model_cognition(self, results: Dict, caption: str, 
                                esn_model, gru_model) -> Dict:
        """
        Comprehensive model brain analysis.
        """
        if not results:
            return {"status": "error", "message": "No predictions available"}
        
        esn_features = esn_model.extract_features(caption)
        esn_state = esn_model.state.copy()
        
        analysis = {
            "esn_analysis": self._analyze_esn_cognition(results["esn"], esn_features, esn_state),
            "gru_analysis": self._analyze_gru_cognition(results["gru"], caption, gru_model),
            "cognitive_comparison": self._compare_cognitive_strategies(
                results["esn"], results["gru"], esn_features
            ),
            "decision_trace": self._trace_decision_path(results, esn_features),
            "confidence_breakdown": self._breakdown_confidence(results)
        }
        
        return analysis
    
    def _analyze_esn_cognition(self, esn_pred: Dict, features: np.ndarray, 
                              state: np.ndarray) -> Dict:
        """Analyze ESN's cognitive process."""
        
        # Feature activation analysis
        feature_names = [
            "caption_length", "spice_level", "ritual_score", "daily_score", "heritage",
            "color_vivid", "texture_words", "serving_style", "has_thali", "has_festival",
            "has_sweet", "has_rice", "veg_focus", "meat_focus", "has_dessert", "has_beverage",
            "has_offering", "has_decoration", "has_people", "has_utensil"
        ]
        
        top_features = []
        for i in np.argsort(features[:20])[::-1][:5]:
            if features[i] > 0.4:
                top_features.append({
                    "feature": feature_names[i],
                    "activation": round(float(features[i]), 3)
                })
        
        # Reservoir state analysis
        state_energy = float(np.linalg.norm(state))
        state_sparsity = float(np.sum(np.abs(state) < 0.1) / len(state))
        
        return {
            "mode": esn_pred["mode"],
            "confidence": round(esn_pred["confidence"], 3),
            "top_features": top_features,
            "reservoir_state": {
                "energy": round(state_energy, 3),
                "sparsity": round(state_sparsity, 3),
                "description": self._describe_reservoir_state(state_energy, state_sparsity)
            },
            "cognitive_style": "Immediate, feature-driven pattern recognition",
            "strengths": ["Fast response", "Direct feature mapping", "Interpretable activations"],
            "limitations": ["No temporal memory", "Limited context integration"]
        }
    
    def _analyze_gru_cognition(self, gru_pred: Dict, caption: str, gru_model) -> Dict:
        """Analyze GRU's cognitive process."""
        
        return {
            "mode": gru_pred["mode"],
            "confidence": round(gru_pred["confidence"], 3),
            "cognitive_style": "Sequential, learned pattern matching",
            "strengths": ["Temporal context", "Learned abstractions", "Robust to noise"],
            "limitations": ["Black box", "Harder to interpret", "Slower training"],
            "decision_basis": "Learned cultural patterns from training data"
        }
    
    def _describe_reservoir_state(self, energy: float, sparsity: float) -> str:
        """Describe ESN reservoir state."""
        if energy > 15 and sparsity < 0.4:
            return "High activation, dense - strong signal detection"
        elif energy > 10:
            return "Moderate activation - clear but not dominant signals"
        elif sparsity > 0.6:
            return "Sparse activation - weak or ambiguous signals"
        else:
            return "Low energy - minimal cultural activation"
    
    def _compare_cognitive_strategies(self, esn_pred: Dict, gru_pred: Dict, 
                                     features: np.ndarray) -> Dict:
        """Compare ESN vs GRU cognitive strategies."""
        
        agreement = esn_pred["mode"] == gru_pred["mode"]
        conf_diff = abs(esn_pred["confidence"] - gru_pred["confidence"])
        
        return {
            "agreement": agreement,
            "confidence_difference": round(conf_diff, 3),
            "esn_strategy": "Bottom-up: immediate feature activation → classification",
            "gru_strategy": "Top-down: learned patterns → classification",
            "interpretation": self._interpret_cognitive_difference(agreement, conf_diff)
        }
    
    def _interpret_cognitive_difference(self, agreement: bool, conf_diff: float) -> str:
        """Interpret cognitive differences."""
        if agreement and conf_diff < 0.2:
            return "Both cognitive strategies converge - stable interpretation"
        elif agreement:
            return "Same classification but different confidence - processing pathway differences"
        else:
            return "Cognitive strategies diverge - ESN sees immediate features differently than GRU's learned patterns"
    
    def _trace_decision_path(self, results: Dict, features: np.ndarray) -> List[str]:
        """Trace the decision-making path."""
        trace = []
        
        trace.append(f"1. Feature extraction: {len(features)} dimensional input")
        trace.append(f"2. ESN reservoir: {features[:5].max():.2f} max activation")
        trace.append(f"3. ESN classification: {results['esn']['mode']} ({results['esn']['confidence']:.2f})")
        trace.append(f"4. GRU sequential: {results['gru']['mode']} ({results['gru']['confidence']:.2f})")
        trace.append(f"5. Final decision: {results['combined_mode']}")
        
        return trace
    
    def _breakdown_confidence(self, results: Dict) -> Dict:
        """Break down confidence sources."""
        esn_conf = results['esn']['confidence']
        gru_conf = results['gru']['confidence']
        
        return {
            "esn_contribution": round(esn_conf, 3),
            "gru_contribution": round(gru_conf, 3),
            "combined_confidence": round(max(esn_conf, gru_conf), 3),
            "confidence_source": "ESN" if esn_conf > gru_conf else "GRU",
            "interpretation": self._interpret_confidence_source(esn_conf, gru_conf)
        }
    
    def _interpret_confidence_source(self, esn_conf: float, gru_conf: float) -> str:
        """Interpret which model drives confidence."""
        if abs(esn_conf - gru_conf) < 0.1:
            return "Balanced confidence from both models"
        elif esn_conf > gru_conf:
            return "ESN drives confidence - strong immediate feature detection"
        else:
            return "GRU drives confidence - learned pattern recognition dominant"