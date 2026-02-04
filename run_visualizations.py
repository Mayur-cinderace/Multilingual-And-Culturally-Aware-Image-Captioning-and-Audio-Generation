# run_all_visualizations.py
# ------------------------------------------------------------
# Complete Visualization Suite: 2D + 3D Reservoir Dynamics
# ------------------------------------------------------------

from cultural_context import CulturalContextManager
from esn_gru_visualization import (
    visualize_esn_activation_flow,
    visualize_cultural_feature_importance,
    visualize_cultural_dimensions,
    visualize_counterfactual_impact,
    visualize_convergence
)
from esn_gru_3d import (
    visualize_3d_esn_reservoir_network,
    visualize_3d_cultural_trajectory,
    visualize_3d_cultural_injection,
    visualize_animated_reservoir_evolution
)

from pathlib import Path

OUTPUT_DIR = Path("viz_server/static")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------
training_samples = [
    # Food Traditional
    ("A traditional thali with dal, rice, roti, and seasonal vegetables", "food_traditional"),
    ("Homemade biryani with fragrant spices and tender meat", "food_traditional"),
    ("Golden crispy jalebi served hot with rabri", "food_traditional"),
    ("Steaming masala dosa with coconut chutney and sambar", "food_traditional"),
    ("Rich paneer tikka marinated in yogurt and spices", "food_traditional"),
    ("Traditional tandoori naan with butter and garlic", "food_traditional"),
    ("Aromatic curry with authentic spices and herbs", "food_traditional"),
    
    # Festival Context
    ("Diwali sweets arranged on a decorative thali with diyas", "festival_context"),
    ("Colorful rangoli designs welcoming guests during Holi", "festival_context"),
    ("Prasad offerings at the temple during Navratri", "festival_context"),
    ("Festive gathering with family celebrating Eid with biryani", "festival_context"),
    ("Traditional puja setup with flowers and incense", "festival_context"),
    ("Decorated sweets for festival celebration with family", "festival_context"),
    ("Sacred offering prepared for religious ceremony", "festival_context"),
    
    # Daily Life
    ("Family dinner with homemade dal and roti", "daily_life"),
    ("Morning chai with breakfast at the kitchen table", "daily_life"),
    ("Lunchbox packed with vegetables and rice for work", "daily_life"),
    ("Evening snacks with samosas and chutney", "daily_life"),
    ("Weekend gathering with friends over tea and pakoras", "daily_life"),
    ("Simple home-cooked meal with family", "daily_life"),
    ("Regular weekday breakfast routine", "daily_life"),
    
    # Generic
    ("A plate of food on a table", "generic"),
    ("Some vegetables in a bowl", "generic"),
    ("Bread and butter", "generic"),
    ("A cup of tea", "generic"),
    ("Restaurant meal", "generic"),
    ("Food item", "generic"),
    ("Meal on a dish", "generic"),
]

# ------------------------------------------------------------
# Initialize and Train Models
# ------------------------------------------------------------
print("=" * 60)
print("🧠 CULTURAL AI VISUALIZATION SUITE")
print("=" * 60)
print("\n🔧 Initializing models...")

cultural_manager = CulturalContextManager()

print("📚 Training with", len(training_samples), "samples...")
cultural_manager.train(training_samples)
print("✅ Models trained successfully!\n")

# ------------------------------------------------------------
# Test Captions
# ------------------------------------------------------------
main_caption = "A traditional thali served during a festive family gathering with aromatic biryani and sweet jalebis"

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

counterfactual_variations = {
    "Generic Framing": "Food on a plate",
    "Festival Emphasis": "A traditional thali served during a festive family gathering with aromatic biryani, sweet jalebis, and decorated with rangoli and diyas",
    "Daily Life Focus": "A thali served during a regular family dinner at home with biryani",
    "Ritual Emphasis": "A sacred offering thali prepared for temple prasad with traditional sweets"
}

# ------------------------------------------------------------
# Generate 2D Visualizations (Original Set)
# ------------------------------------------------------------
print("=" * 60)
print("📊 GENERATING 2D VISUALIZATIONS")
print("=" * 60)

print("\n1️⃣ Activation Flow...")
visualize_esn_activation_flow(
    caption=main_caption,
    esn=cultural_manager.esn,
    steps=15,
    top_neurons=50,
    output_path=OUTPUT_DIR / "activation_flow.html"
)
print("✅ activation_flow.html")

print("\n2️⃣ Feature Importance...")
visualize_cultural_feature_importance(
    caption=main_caption,
    esn=cultural_manager.esn,
    gru=cultural_manager.gru,
    output_path=OUTPUT_DIR / "feature_importance.html"
)
print("✅ feature_importance.html")

print("\n3️⃣ Cultural Dimensions...")
visualize_cultural_dimensions(
    captions=diverse_captions,
    esn=cultural_manager.esn,
    output_path=OUTPUT_DIR / "cultural_space.html"
)
print("✅ cultural_space.html")

print("\n4️⃣ Counterfactual Analysis...")
visualize_counterfactual_impact(
    original_caption=main_caption,
    modified_captions=counterfactual_variations,
    esn=cultural_manager.esn,
    output_path=OUTPUT_DIR / "counterfactual.html"
)
print("✅ counterfactual.html")

print("\n5️⃣ Convergence Dynamics...")
visualize_convergence(
    caption=main_caption,
    esn=cultural_manager.esn,
    steps=50,
    output_path=OUTPUT_DIR / "convergence.html"
)
print("✅ convergence.html")

# ------------------------------------------------------------
# Generate 3D Visualizations (NEW!)
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("🎨 GENERATING 3D RESERVOIR DYNAMICS")
print("=" * 60)

print("\n6️⃣ 3D ESN Reservoir Network...")
visualize_3d_esn_reservoir_network(
    caption=main_caption,
    esn=cultural_manager.esn,
    steps=10,
    sample_neurons=100,
    output_path=OUTPUT_DIR / "esn_reservoir_3d.html"
)
print("✅ esn_reservoir_3d.html")

print("\n7️⃣ 3D Cultural Trajectory (ESN vs GRU)...")
visualize_3d_cultural_trajectory(
    caption=main_caption,
    esn=cultural_manager.esn,
    gru=cultural_manager.gru,
    steps=20,
    output_path=OUTPUT_DIR / "cultural_trajectory_3d.html"
)
print("✅ cultural_trajectory_3d.html")

print("\n8️⃣ 3D Cultural Injection (Data Flow)...")
visualize_3d_cultural_injection(
    caption=main_caption,
    esn=cultural_manager.esn,
    gru=cultural_manager.gru,
    output_path=OUTPUT_DIR / "cultural_injection_3d.html"
)
print("✅ cultural_injection_3d.html")

print("\n9️⃣ Animated Reservoir Evolution...")
visualize_animated_reservoir_evolution(
    caption=main_caption,
    esn=cultural_manager.esn,
    steps=20,
    sample_neurons=50,
    output_path=OUTPUT_DIR / "reservoir_evolution_animated.html"
)
print("✅ reservoir_evolution_animated.html")

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("🎉 ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("=" * 60)

print("\n📊 Generated Files:")
print("   2D Visualizations:")
print("   ├── activation_flow.html")
print("   ├── feature_importance.html")
print("   ├── cultural_space.html")
print("   ├── counterfactual.html")
print("   └── convergence.html")
print("\n   3D Reservoir Dynamics:")
print("   ├── esn_reservoir_3d.html")
print("   ├── cultural_trajectory_3d.html")
print("   ├── cultural_injection_3d.html")
print("   └── reservoir_evolution_animated.html")

print(f"\n📁 Location: {OUTPUT_DIR.absolute()}")

print("\n🌐 To view:")
print("   1. cd viz_server")
print("   2. python server.py")
print("   3. Open: http://localhost:8081")

print("\n" + "=" * 60)