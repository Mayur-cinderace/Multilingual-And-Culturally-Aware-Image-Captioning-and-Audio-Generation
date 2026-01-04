# hashtag_generator.py (Updated)
import re
import random

def extract_candidate_phrases(text: str):
    """
    Extract salient phrases without any cultural vocabulary.
    Uses simple, explainable heuristics.
    """
    text = text.lower()

    # Remove stop-like boilerplate
    text = re.sub(r'\b(the|a|an|with|on|in|of|and)\b', '', text)

    # Extract noun-like chunks (no POS tagger)
    tokens = re.findall(r"[a-z]{4,}", text)

    # Heuristic: later words tend to be more specific
    weighted = []
    for i, t in enumerate(tokens):
        weight = (i + 1) / len(tokens)
        weighted.append((t, weight))

    # Sort by salience
    weighted.sort(key=lambda x: x[1], reverse=True)

    return [w for w, _ in weighted[:5]]


def stylistic_transform(word: str):
    """
    Turn a concept into a hashtag style without semantics.
    """
    styles = [
        lambda w: w.capitalize(),
        lambda w: w.capitalize() + "Vibes",
        lambda w: w.capitalize() + "Moments",
        lambda w: "My" + w.capitalize(),
        lambda w: w.capitalize() + "Story",
        lambda w: w.capitalize() + "Life",  # New
        lambda w: "Explore" + w.capitalize(),  # New
    ]
    return "#" + random.choice(styles)(word)


def generate_hashtags(caption: str, min_tags: int = 2):
    candidates = extract_candidate_phrases(caption)

    hashtags = []
    for c in candidates:
        hashtags.append(stylistic_transform(c))
        if len(hashtags) >= min_tags:
            break

    # Absolute fallback (still generic)
    while len(hashtags) < min_tags:
        hashtags.append(random.choice([
            "#VisualStory",
            "#EverydayMoments",
            "#CapturedScene",
            "#SceneVibes"  # New
        ]))

    # Deduplicate, preserve order
    hashtags = list(dict.fromkeys(hashtags))[:min_tags]

    return "  ".join(hashtags)
