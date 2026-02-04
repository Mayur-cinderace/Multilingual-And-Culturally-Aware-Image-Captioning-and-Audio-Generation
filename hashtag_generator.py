# hashtag_generator.py (Production Quality - v3.0)
import re
from collections import Counter
import string

# Expanded stopwords and noise terms
STOPWORDS = {
    'the', 'a', 'an', 'with', 'and', 'or', 'of', 'in', 'on', 'at', 
    'to', 'for', 'from', 'this', 'that', 'is', 'are', 'was', 'were',
    'showing', 'shows', 'featuring', 'features', 'image', 'picture',
    'there', 'here', 'where', 'when', 'what', 'which', 'who', 'how',
    'has', 'have', 'had', 'been', 'being', 'also', 'some', 'such',
    'can', 'could', 'will', 'would', 'should', 'may', 'might',
    'its', 'their', 'them', 'these', 'those'
}

# Visual noise terms that shouldn't be hashtags
VISUAL_NOISE = {
    'watermark', 'logo', 'text', 'writing', 'background',
    'foreground', 'blur', 'blurred', 'visible', 'shown',
    'displayed', 'placed', 'positioned', 'located'
}

# Minimum quality thresholds
MIN_WORD_LENGTH = 4
MIN_WORD_FREQUENCY_FOR_COMPOUND = 2


def is_meaningful_word(word: str) -> bool:
    """
    Check if word is meaningful for hashtags.
    Filters out noise, stopwords, and low-quality terms.
    """
    if not word or len(word) < MIN_WORD_LENGTH:
        return False
    
    if word in STOPWORDS or word in VISUAL_NOISE:
        return False
    
    # Reject if mostly numbers
    if sum(c.isdigit() for c in word) > len(word) / 2:
        return False
    
    # Reject if contains special characters
    if not word.isalpha():
        return False
    
    # Reject common verb forms that don't make good hashtags
    if word.endswith(('ing', 'ed')) and len(word) <= 6:
        return False
    
    return True


def extract_noun_phrases(text: str) -> list:
    """
    Extract potential noun phrases using simple heuristics.
    Focus on content words that represent objects, actions, or concepts.
    """
    # Clean text
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Split into words
    words = text.split()
    
    # Filter meaningful words
    meaningful = [w for w in words if is_meaningful_word(w)]
    
    if not meaningful:
        return []
    
    # Extract 2-word phrases (common noun phrases)
    phrases = []
    for i in range(len(meaningful) - 1):
        phrase = f"{meaningful[i]} {meaningful[i+1]}"
        # Only keep if both words are substantial
        if len(meaningful[i]) >= 5 and len(meaningful[i+1]) >= 5:
            phrases.append(phrase)
    
    return phrases


def score_term_salience(word: str, position: int, total_words: int, 
                        frequency: int, max_freq: int) -> float:
    """
    Multi-factor scoring for term importance.
    """
    # Position weight (middle and end are more important)
    if position < 3:
        position_score = 0.3  # Penalize generic intro words
    elif position > total_words * 0.7:
        position_score = 1.0  # End words often most specific
    else:
        position_score = 0.7  # Middle words
    
    # Frequency weight (but not too common)
    freq_ratio = frequency / max_freq
    if freq_ratio > 0.5:
        freq_score = 0.8  # Very common = likely generic
    else:
        freq_score = 1.0
    
    # Length weight (longer = more specific)
    length_score = min(len(word) / 10.0, 1.0)
    
    # Composite score
    score = (position_score * 0.4 + freq_score * 0.3 + length_score * 0.3)
    
    return score


def extract_salient_terms(caption: str, max_terms=10) -> list:
    """
    Extract meaningful, salient terms with quality filtering.
    """
    if not caption or len(caption.strip()) < 5:
        return []
    
    text = caption.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    words = text.split()
    words = [w for w in words if is_meaningful_word(w)]
    
    if not words:
        return []
    
    # Frequency analysis
    freq = Counter(words)
    max_freq = max(freq.values()) if freq else 1
    
    # Score each unique word
    scored = []
    seen_positions = {}
    
    for i, word in enumerate(words):
        if word not in seen_positions:
            seen_positions[word] = i
            score = score_term_salience(word, i, len(words), freq[word], max_freq)
            scored.append((word, score))
    
    # Sort by score
    scored.sort(key=lambda x: -x[1])
    
    # Return top terms
    return [word for word, _ in scored[:max_terms]]


def generate_creative_variations(base_term: str) -> list:
    """
    Generate tasteful variations of a base term.
    Avoids generic suffixes, focuses on meaningful combinations.
    """
    variations = []
    
    # Capitalize base
    base_cap = base_term.capitalize()
    variations.append(base_cap)
    
    # Meaningful suffixes only
    quality_suffixes = {
        'Life': 0.8,
        'Culture': 0.9,
        'Story': 0.7,
        'Journey': 0.6,
        'Art': 0.8,
        'Scene': 0.5
    }
    
    # Only add suffix if base term is substantial (≥6 chars)
    if len(base_term) >= 6:
        for suffix, prob in quality_suffixes.items():
            if len(variations) < 3:  # Limit variations
                variations.append(f"{base_cap}{suffix}")
    
    return variations


def generate_hashtags(caption: str, min_tags=3, max_tags=5) -> str:
    """
    Generate high-quality, contextual hashtags.
    
    Quality guarantees:
    - No stopwords or noise terms
    - No generic visual descriptions
    - Meaningful content words only
    - Diverse and relevant
    """
    
    if not caption or len(caption.strip()) < 5:
        return "#Captured #Moment #Scene"
    
    hashtags = []
    
    # Extract salient terms
    terms = extract_salient_terms(caption, max_terms=12)
    
    if not terms:
        return "#Captured #Moment #Scene"
    
    # Strategy 1: Top 2 direct terms (most salient)
    for term in terms[:2]:
        hashtags.append(f"#{term.capitalize()}")
    
    # Strategy 2: Noun phrases (if available)
    phrases = extract_noun_phrases(caption)
    if phrases:
        # Take best phrase (first one, usually most relevant)
        phrase = phrases[0]
        phrase_tag = ''.join(word.capitalize() for word in phrase.split())
        if phrase_tag not in [h.replace('#', '') for h in hashtags]:
            hashtags.append(f"#{phrase_tag}")
    
    # Strategy 3: Creative variation of top term
    if len(terms) > 0 and len(hashtags) < max_tags:
        variations = generate_creative_variations(terms[0])
        for var in variations:
            tag = f"#{var}"
            if tag not in hashtags and len(hashtags) < max_tags:
                hashtags.append(tag)
    
    # Strategy 4: Secondary term if still need more
    if len(terms) > 2 and len(hashtags) < max_tags:
        for term in terms[2:4]:
            tag = f"#{term.capitalize()}"
            if tag not in hashtags:
                hashtags.append(tag)
                if len(hashtags) >= max_tags:
                    break
    
    # Ensure minimum count with quality fallbacks
    quality_fallbacks = [
        '#CulturalMoment',
        '#HeritageScene', 
        '#AuthenticLife',
        '#VisualStory',
        '#CapturedCulture'
    ]
    
    fallback_idx = 0
    while len(hashtags) < min_tags and fallback_idx < len(quality_fallbacks):
        fallback = quality_fallbacks[fallback_idx]
        if fallback not in hashtags:
            hashtags.append(fallback)
        fallback_idx += 1
    
    # Final quality check: remove any that slipped through
    hashtags = [
        tag for tag in hashtags 
        if not any(noise in tag.lower() for noise in VISUAL_NOISE)
        and len(tag) > 3  # Minimum length
    ]
    
    # Limit to max
    hashtags = hashtags[:max_tags]
    
    # Deduplicate case-insensitively
    seen = set()
    unique_hashtags = []
    for tag in hashtags:
        tag_lower = tag.lower()
        if tag_lower not in seen:
            unique_hashtags.append(tag)
            seen.add(tag_lower)
    
    return "  ".join(unique_hashtags)

