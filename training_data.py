# training_data.py
import numpy as np

training_samples = [
    # food_traditional  ── 10 examples
    ("A bowl of spicy curry garnished with fresh herbs.", "food_traditional"),
    ("Aromatic masala dish served in a traditional bowl.", "food_traditional"),
    ("Gravy with spices and vegetables.", "food_traditional"),
    ("Spiced rice garnished with nuts.", "food_traditional"),
    ("Traditional meal with rich flavors.", "food_traditional"),
    ("Creamy paneer butter masala with garlic naan.", "food_traditional"),
    ("Dal tadka served with jeera rice.", "food_traditional"),
    ("Vegetable korma in coconut gravy.", "food_traditional"),
    ("Chicken tikka masala with butter naan.", "food_traditional"),
    ("Aloo gobi dry sabzi with roti.", "food_traditional"),

    # festival_context  ── 8 examples
    ("People celebrating with lights and sweets during festival.", "festival_context"),
    ("Ritual offerings in a celebration.", "festival_context"),
    ("Festival gathering with colorful decorations.", "festival_context"),
    ("Traditional dance during celebratory event.", "festival_context"),
    ("Diwali diyas and rangoli at home entrance.", "festival_context"),
    ("Holi colors thrown in joyful gathering.", "festival_context"),
    ("Ganesh idol decorated with flowers and modak.", "festival_context"),
    ("Navratri garba dance in traditional clothes.", "festival_context"),

    # daily_life  ── 4 examples (smaller because it's closer to generic)
    ("Family having a simple home meal.", "daily_life"),
    ("Daily routine with breakfast in bowl.", "daily_life"),
    ("Home-cooked dish served to family.", "daily_life"),
    ("Everyday gathering around the table.", "daily_life"),

    # generic  ── 18 examples  (≈45%)
    ("A landscape with mountains and sky.", "generic"),
    ("Abstract art piece on wall.", "generic"),
    ("City street with cars.", "generic"),
    ("Book on a table.", "generic"),
    ("Random object in room.", "generic"),
    ("Sunset over calm ocean.", "generic"),
    ("Modern office building exterior.", "generic"),
    ("Person walking in park.", "generic"),
    ("Highway traffic at dusk.", "generic"),
    ("Clouds in blue sky.", "generic"),
    ("Cat sleeping on windowsill.", "generic"),
    ("Laptop and notebook on desk.", "generic"),
    ("Train passing through countryside.", "generic"),
    ("Flower pot on balcony.", "generic"),
    ("Snow-covered trees in winter.", "generic"),
    ("Empty beach at sunrise.", "generic"),
    ("Street lamp at night.", "generic"),
    ("Bicycle parked near wall.", "generic"),
]

def augment_samples(samples):
    augmented = []
    templates = [
        "An image showing {}",
        "A scene with {}",
        "A close-up of {}",
        "A casual view of {}",
    ]

    for caption, mode in samples:
        augmented.append((caption, mode))
        for t in templates:
            augmented.append((t.format(caption.lower()), mode))
    return augmented

augmented_samples = augment_samples(training_samples)
