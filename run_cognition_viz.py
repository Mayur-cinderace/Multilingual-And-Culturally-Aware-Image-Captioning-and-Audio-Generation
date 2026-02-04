# run_cognition_viz.py
# ------------------------------------------------------------
# Complete ESN / GRU Cognition Visualization Runner
# ------------------------------------------------------------

from cultural_context import CulturalContextManager
from esn_gru_visualization import (
    visualize_esn_activation_flow,
    visualize_cultural_feature_importance,
    visualize_cultural_dimensions,
    visualize_counterfactual_impact,
    visualize_convergence
)

from pathlib import Path

OUTPUT_DIR = Path("viz_server/static")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Setup model
# ------------------------------------------------------------
cultural_manager = CulturalContextManager()

# ⚠️ IMPORTANT: train before visualizing
# Training samples for cultural context
training_samples = [
    # Food Traditional
    ("A traditional thali with dal, rice, roti, and seasonal vegetables", "food_traditional"),
    ("Homemade biryani with fragrant spices and tender meat", "food_traditional"),
    ("Golden crispy jalebi served hot with rabri", "food_traditional"),
    ("Steaming masala dosa with coconut chutney and sambar", "food_traditional"),
    ("Rich paneer tikka marinated in yogurt and spices", "food_traditional"),
    
    # Festival Context
    ("Diwali sweets arranged on a decorative thali with diyas", "festival_context"),
    ("Colorful rangoli designs welcoming guests during Holi", "festival_context"),
    ("Prasad offerings at the temple during Navratri", "festival_context"),
    ("Festive gathering with family celebrating Eid with biryani", "festival_context"),
    ("Traditional puja setup with flowers and incense", "festival_context"),
    
    # Daily Life
    ("Family dinner with homemade dal and roti", "daily_life"),
    ("Morning chai with breakfast at the kitchen table", "daily_life"),
    ("Lunchbox packed with vegetables and rice for work", "daily_life"),
    ("Evening snacks with samosas and chutney", "daily_life"),
    ("Weekend gathering with friends over tea and pakoras", "daily_life"),
    
    # Generic
    ("A plate of food on a table", "generic"),
    ("Some vegetables in a bowl", "generic"),
    ("Bread and butter", "generic"),
    ("A cup of tea", "generic"),
    ("Restaurant meal", "generic"),
]

print("🔧 Training cultural models...")
cultural_manager.train(training_samples)
print("✅ Models trained successfully!\n")

# ------------------------------------------------------------
# Test captions for different visualizations
# ------------------------------------------------------------

# Main caption for detailed analysis
main_caption = "A traditional thali served during a festive family gathering with aromatic biryani and sweet jalebis"

# Multiple captions for cultural space visualization
diverse_captions = [
    "Traditional wedding feast with elaborate thali",
    "Diwali celebration with homemade sweets and diyas",
    "Simple family dinner with dal and roti",
    "Festival prasad distributed at temple",
    "Morning chai and breakfast routine",
    "A plate of pasta on a table",
    "Colorful rangoli welcoming guests during Holi",
    "Weekend gathering with friends and snacks",
]

# Counterfactual variations
counterfactual_variations = {
    "Generic Framing": "Food on a plate",
    "Festival Emphasis": "A traditional thali served during a festive family gathering with aromatic biryani, sweet jalebis, and decorated with rangoli and diyas",
    "Daily Life Focus": "A thali served during a regular family dinner at home with biryani",
    "Ritual Emphasis": "A sacred offering thali prepared for temple prasad with traditional sweets"
}

# ------------------------------------------------------------
# Generate all visualizations
# ------------------------------------------------------------

print("📊 Generating Visualization 1: Activation Flow...")
visualize_esn_activation_flow(
    caption=main_caption,
    esn=cultural_manager.esn,
    steps=15,
    top_neurons=50,
    output_path=OUTPUT_DIR / "activation_flow.html"
)
print("✅ Activation flow complete\n")

print("🎯 Generating Visualization 2: Feature Importance...")
visualize_cultural_feature_importance(
    caption=main_caption,
    esn=cultural_manager.esn,
    gru=cultural_manager.gru,
    output_path=OUTPUT_DIR / "feature_importance.html"
)
print("✅ Feature importance complete\n")

print("🗺️ Generating Visualization 3: Cultural Dimensions...")
visualize_cultural_dimensions(
    captions=diverse_captions,
    esn=cultural_manager.esn,
    output_path=OUTPUT_DIR / "cultural_space.html"
)
print("✅ Cultural space complete\n")

print("🔄 Generating Visualization 4: Counterfactual Analysis...")
visualize_counterfactual_impact(
    original_caption=main_caption,
    modified_captions=counterfactual_variations,
    esn=cultural_manager.esn,
    output_path=OUTPUT_DIR / "counterfactual.html"
)
print("✅ Counterfactual analysis complete\n")

print("📈 Generating Visualization 5: Convergence Dynamics...")
visualize_convergence(
    caption=main_caption,
    esn=cultural_manager.esn,
    steps=50,
    output_path=OUTPUT_DIR / "convergence.html"
)
print("✅ Convergence analysis complete\n")

print("=" * 60)
print("🎉 All visualizations generated successfully!")
print("=" * 60)
print("\n📁 Output directory:", OUTPUT_DIR.absolute())
print("\n🌐 To view the visualizations:")
print("   1. Navigate to viz_server directory")
print("   2. Run: python server.py")
print("   3. Open: http://localhost:8081")
print("\n" + "=" * 60)