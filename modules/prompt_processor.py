"""
modules/prompt_processor.py
---------------------------
Prompt engineering + content safety + poster copy generation for WellAware AI.

Responsibilities:
1. Validate user inputs (blocked terms, claim terms).
2. Clean and engineer AI prompts for clean, isolated vector artwork with no messy backgrounds.
3. Generate relevant, professional healthcare poster copy (Headline, Slogan, Body, Bullet Tips, Quote, Org).
4. Provide category presets and prompt suggestions matching reference posters.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

import config


# ------------------------------------------------------------------
# Data containers
# ------------------------------------------------------------------
@dataclass
class ValidationResult:
    """Outcome of validating user input."""
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class PromptBundle:
    """Everything the image generator needs, plus human-readable summary."""
    positive: str
    negative: str
    category: str
    style: str
    user_text: str
    readable: str


@dataclass
class PosterCopy:
    """Structured, professional content for healthcare campaign posters."""
    headline: str
    slogan: str
    body: str
    bullet_points: List[str]
    quote: str
    org_name: str
    helpline: str


# ------------------------------------------------------------------
# Campaign copy generation & presets
# ------------------------------------------------------------------
def get_campaign_preset(category: str, custom_category: str = "") -> PosterCopy:
    """Return default professional poster copy matching the selected category."""
    cat_key = custom_category.strip() if category == config.CUSTOM_CATEGORY and custom_category.strip() else category
    preset = config.CAMPAIGN_PRESETS.get(cat_key)
    
    if not preset:
        # Fallback for custom or unknown category
        title = cat_key.upper() if cat_key else "HEALTH & WELLNESS"
        preset = {
            "headline": title,
            "slogan": "Empowering Healthy Living, Mind, and Body",
            "body": "Everyday wellbeing can include enjoyable movement, varied meals, restful moments, and supportive relationships.",
            "bullet_points": [
                "Prioritize balanced nutrition and daily hydration",
                "Choose comfortable movement that you enjoy",
                "Care for mental well-being and get restful sleep",
                "Support and encourage your family and community",
            ],
            "quote": "Wellness is not a destination, but a continuous journey of positive daily choices.",
            "org_name": "Your Organization Name",
            "helpline": "Your helpline / website here",
            "artwork_prompt": f"clean modern illustration representing {cat_key.lower() or 'wellness'}",
        }

    return PosterCopy(
        headline=preset["headline"],
        slogan=preset["slogan"],
        body=preset["body"],
        bullet_points=list(preset["bullet_points"]),
        quote=preset["quote"],
        org_name=preset["org_name"],
        helpline=preset["helpline"],
    )


HUMAN_KEYWORDS = [
    "woman", "women", "lady", "girl", "female",
    "man", "men", "guy", "boy", "male",
    "person", "people", "human", "face", "portrait",
    "doctor", "nurse", "physician", "optometrist", "dentist", "surgeon",
    "patient", "child", "children", "baby", "infant", "kid", "kids",
    "senior", "elder", "elderly", "grandparent", "mother", "father",
    "adult", "adults", "professional",
]

POSITIVE_PROMPT_BANNED_WORDS = (
    "no", "without", "not", "avoid", "poster", "posters", "dribbble", "banner",
    "banners", "label", "labels", "sign", "signs", "chart", "charts", "infographic",
    "typography", "logo", "logos", "vision chart",
    "healthcare graphic", "award-winning design", "text", "lettering", "letters",
    "words", "writing", "caption", "slogan",
)


def sanitize_positive_prompt(text: str) -> str:
    """Remove negation and lettering-oriented terms before text enters the positive prompt."""
    sanitized = re.sub(r"\s+", " ", text or "").strip()
    for term in sorted(POSITIVE_PROMPT_BANNED_WORDS, key=len, reverse=True):
        sanitized = re.sub(
            r"(?<!\w)" + re.escape(term).replace(r"\ ", r"\s+") + r"(?!\w)",
            " ",
            sanitized,
            flags=re.IGNORECASE,
        )
    return re.sub(r"\s+", " ", sanitized).strip(" ,.;:-")


def lint_poster_copy(copy_or_text: PosterCopy | str) -> List[str]:
    """Return risky claim phrases found in editable poster copy."""
    if isinstance(copy_or_text, PosterCopy):
        text = " ".join(
            [copy_or_text.headline, copy_or_text.slogan, copy_or_text.body,
             *copy_or_text.bullet_points, copy_or_text.quote,
             copy_or_text.org_name, copy_or_text.helpline]
        )
    else:
        text = copy_or_text
    return _find_terms(text, config.CLAIM_TERMS)


@lru_cache(maxsize=1)
def _get_clip_tokenizer():
    from transformers import CLIPTokenizer
    return CLIPTokenizer.from_pretrained(config.CLIP_TOKENIZER_ID)


def count_clip_tokens(text: str, tokenizer=None) -> int:
    """Count prompt tokens using CLIP's tokenizer, loading it lazily for the UI."""
    if tokenizer is None:
        tokenizer = _get_clip_tokenizer()
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        return len(backend.encode(text, add_special_tokens=False).ids)
    tokenize = getattr(tokenizer, "tokenize", None)
    if callable(tokenize):
        return len(tokenize(text))
    return len(tokenizer.encode(text, add_special_tokens=False))


def truncate_prompt_by_priority(
    subject: str,
    style: str = "",
    palette: str = "",
    extras: str = "",
    tokenizer=None,
    max_tokens: int = 75,
) -> str:
    """Keep the subject intact and spend remaining CLIP tokens on lower-priority details."""
    parts = [sanitize_positive_prompt(subject)]
    if count_clip_tokens(parts[0], tokenizer) > max_tokens:
        return parts[0]

    for candidate in (style, palette):
        clean = sanitize_positive_prompt(candidate)
        if not clean:
            continue
        fitted_words = []
        for word in clean.split():
            candidate_text = " ".join(fitted_words + [word])
            proposed = ", ".join(parts + [candidate_text])
            if count_clip_tokens(proposed, tokenizer) > max_tokens:
                break
            fitted_words.append(word)
        if fitted_words:
            parts.append(" ".join(fitted_words))

    clean_extras = sanitize_positive_prompt(extras)
    if clean_extras:
        proposed = ", ".join(parts + [clean_extras])
        if count_clip_tokens(proposed, tokenizer) <= max_tokens:
            parts.append(clean_extras)
    return ", ".join(parts)


def truncate_negative_prompt(text: str, tokenizer=None, max_tokens: int = 75) -> str:
    """Keep the highest-priority negative terms within CLIP's prompt context."""
    phrases = [phrase.strip() for phrase in text.split(",") if phrase.strip()]
    kept = []
    for phrase in phrases:
        candidate = ", ".join(kept + [phrase])
        if count_clip_tokens(candidate, tokenizer) > max_tokens:
            break
        kept.append(phrase)
    return ", ".join(kept)


def generate_poster_copy(prompt: str, category: str, custom_category: str = "") -> PosterCopy:
    """
    Intelligently generate relevant poster copy based on both the user's prompt
    and the campaign category with 100% accurate spelling and actionable advice.
    """
    base = get_campaign_preset(category, custom_category)
    combined = f"{prompt or ''} {category or ''} {custom_category or ''}".lower()
    cleaned = combined

    # Specialized health campaign topics:
    if any(k in combined for k in ["eye", "vision", "optometr", "sight", "cataract", "glaucoma", "blind"]):
        return PosterCopy(
            headline="EYE WELLNESS",
            slogan="Make room for comfortable screen breaks",
            body="Create a comfortable space for reading, screen breaks, and enjoying the world around you.",
            bullet_points=[
                "Pause from screens when convenient",
                "Use comfortable lighting when reading",
                "Enjoy time outdoors in a way that feels right",
                "Ask a qualified professional about personal questions",
            ],
            quote="Small moments of care can make everyday life feel more balanced.",
            org_name="Your Organization Name",
            helpline="Your helpline / website here",
        )
    elif any(k in combined for k in ["dental", "tooth", "teeth", "oral", "gum", "dentist"]):
        return PosterCopy(
            headline="EVERYDAY ORAL WELLNESS",
            slogan="Make room for everyday oral-care routines",
            body="A relaxed daily routine can include brushing, caring for your smile, and asking questions when you need guidance.",
            bullet_points=[
                "Build a routine that works for your day",
                "Keep oral-care items in a convenient place",
                "Make time for breaks and water",
                "Ask a qualified dental professional for personal guidance",
            ],
            quote="A genuine smile is the shortest distance between two people.",
            org_name="Your Organization Name",
            helpline="Your helpline / website here",
        )
    elif any(k in combined for k in ["dengue", "malaria", "mosquito", "vector", "chikungunya"]):
        return PosterCopy(
            headline="COMMUNITY WELLNESS",
            slogan="Clean Surroundings, Safe and Healthy Communities",
            body="A clean, comfortable shared space can support everyday wellbeing and community connection.",
            bullet_points=[
                "Keep shared spaces tidy and comfortable",
                "Spend time together on simple neighborhood activities",
                "Make room for rest and enjoyable outdoor time",
                "Ask qualified local professionals for personal guidance",
            ],
            quote="Kindness and community support make a meaningful difference.",
            org_name="Your Organization Name",
            helpline="Your helpline / website here",
        )
    elif any(k in combined for k in ["fruit", "vegetable", "woman", "salad", "nutrition", "diet", "food", "eat"]):
        return PosterCopy(
            headline="FRESH NOURISHMENT, VIBRANT LIFE",
            slogan="Nourish Your Body from the Inside Out",
            body="Make everyday meals colorful with a variety of fruits, vegetables, grains, and other familiar foods.",
            bullet_points=[
                "Try a variety of colors and textures in meals",
                "Keep water nearby throughout the day",
                "Share meal preparation with family or friends",
                "Enjoy meals at a comfortable pace",
            ],
            quote="A colorful meal can be a joyful part of the day.",
            org_name="Your Organization Name",
            helpline="Your helpline / website here",
        )
    elif any(k in combined for k in ["pediatric", "vaccin", "immuniz", "baby", "infant", "child", "children"]):
        return PosterCopy(
            headline="NURTURING CHILD WELLNESS",
            slogan="Healthy Steps for Little Champions",
            body="Make time for play, rest, nourishing meals, and supportive moments together.",
            bullet_points=[
                "Make space for joyful play and creative activities",
                "Enjoy a variety of foods together",
                "Build in quiet moments for rest",
                "Encourage kind conversations and connection",
            ],
            quote="Children are our greatest treasure and the future of our communities.",
            org_name="Your Organization Name",
            helpline="Your helpline / website here",
        )
    elif any(k in combined for k in ["elder", "senior", "grandparent", "aging", "stroke", "mobility", "orthoped", "joint"]):
        return PosterCopy(
            headline="ACTIVE & HEALTHY AGING",
            slogan="Vitality, Wisdom, and Grace at Every Stage of Life",
            body="Everyday routines can include comfortable movement, meaningful conversations, and moments of rest.",
            bullet_points=[
                "Choose comfortable movement that suits you",
                "Stay connected with friends, family, or community",
                "Enjoy a variety of foods and restful moments",
                "Ask a qualified professional about personal wellness needs",
            ],
            quote="Age is an honor and a testament to a life lived with purpose and care.",
            org_name="Your Organization Name",
            helpline="Your helpline / website here",
        )
    elif any(k in combined for k in ["doctor", "nurse", "hospital", "clinic", "scrub", "surgeon"]):
        return PosterCopy(
            headline="HEALTHCARE HEROES IN ACTION",
            slogan="Compassionate Care, Dedicated to Every Patient",
            body="Recognize the people who help build supportive and welcoming communities.",
            bullet_points=[
                "Share appreciation and encouragement",
                "Make time for rest and supportive conversations",
                "Explore local community activities",
                "Choose routines that feel comfortable for you",
            ],
            quote="Dedicated hands and caring hearts build the foundation of community wellness.",
            org_name="Your Organization Name",
            helpline=base.helpline,
        )
    elif any(k in combined for k in ["skin", "derma", "sunscreen", "uv"]):
        return PosterCopy(
            headline="RADIANT SKIN, HEALTHY LIVING",
            slogan="Enjoy moments outdoors in ways that feel comfortable",
            body="Make time for comfortable routines, rest, and moments outdoors that suit you.",
            bullet_points=[
                "Choose comfortable clothing for the weather",
                "Keep water nearby throughout the day",
                "Enjoy a gentle outdoor activity",
                "Ask a qualified professional about personal questions",
            ],
            quote="Healthy skin is a reflection of overall wellness and mindful daily care.",
            org_name="Your Organization Name",
            helpline="Your helpline / website here",
        )
    elif any(k in combined for k in ["kidney", "renal", "dialysis"]):
        return PosterCopy(
            headline="EVERYDAY WELLNESS",
            slogan="Make room for routines that feel right for you",
            body="Explore everyday routines that include varied meals, enjoyable movement, rest, and connection.",
            bullet_points=[
                "Enjoy a variety of familiar foods",
                "Choose comfortable movement you enjoy",
                "Make time for rest and supportive connections",
                "Ask a qualified professional about personal questions",
            ],
            quote="Small moments of care can make everyday life feel more balanced.",
            org_name="Your Organization Name",
            helpline="Your helpline / website here",
        )
    elif any(k in combined for k in ["yoga", "meditat", "peace", "calm", "relax", "mindfulness"]):
        return PosterCopy(
            headline="MINDFUL LIVING & INNER PEACE",
            slogan="Breathe Deeply, Move Mindfully, Live Fully",
            body="Make room for quiet moments, supportive conversations, and activities you enjoy.",
            bullet_points=[
                "Try a calming activity that you enjoy",
                "Make time for supportive conversations",
                "Spend time outdoors when it suits you",
                "Build quiet moments into your routine",
            ],
            quote="Peace of mind is not the absence of storms, but calmness within the soul.",
            org_name="Your Organization Name",
            helpline=base.helpline,
        )

    # Built-in campaign presets check
    if category != config.CUSTOM_CATEGORY and category in config.CAMPAIGN_PRESETS and category != "General Wellness":
        return PosterCopy(
            headline=base.headline,
            slogan=base.slogan,
            body=base.body,
            bullet_points=list(base.bullet_points),
            quote=base.quote,
            org_name=base.org_name,
            helpline=base.helpline,
        )

    # Inspect prompt keywords to dynamically find best preset
    if any(k in cleaned for k in ["blood", "donor", "donation"]):
        preset = config.CAMPAIGN_PRESETS["Blood Donation Awareness"]
    elif any(k in cleaned for k in ["cancer", "ribbon", "tumor", "oncolog"]):
        preset = config.CAMPAIGN_PRESETS["Cancer Awareness"]
    elif any(k in cleaned for k in ["diabet", "sugar", "glucose", "insulin"]):
        preset = config.CAMPAIGN_PRESETS["Diabetes Awareness"]
    elif any(k in cleaned for k in ["run", "fitness", "exercise", "workout", "gym", "sport", "athletic", "movement"]):
        preset = config.CAMPAIGN_PRESETS["Physical Fitness"]
    elif any(k in cleaned for k in ["women", "female", "maternal", "breast"]):
        preset = config.CAMPAIGN_PRESETS["Women's Health Awareness"]
    elif any(k in cleaned for k in ["mental", "mind", "brain", "stress", "inside"]):
        preset = config.CAMPAIGN_PRESETS["Mental Health"]
    elif any(k in cleaned for k in ["heart", "lung", "respiratory", "cardio", "pulse"]):
        preset = config.CAMPAIGN_PRESETS["Heart Health"]
    elif any(k in cleaned for k in ["world health", "community"]):
        preset = config.CAMPAIGN_PRESETS["World Health Day"]
    else:
        # Dynamic universal health campaign generator for ANY custom topic
        topic_title = custom_category.strip() or category
        if topic_title in ("General Wellness", config.CUSTOM_CATEGORY, ""):
            # Extract main noun phrase from prompt if possible
            words = [w for w in re.findall(r"[a-zA-Z]+", prompt) if len(w) > 3 and w.lower() not in {"poster", "camp", "with", "make", "draw", "generate", "create", "about"}]
            topic_title = " ".join(words[:3]).title() if words else "Health & Wellness"

        return PosterCopy(
            headline=topic_title.upper(),
            slogan="Small moments of care and connection",
            body="Wellness can include enjoyable movement, varied meals, restful moments, and supportive relationships.",
            bullet_points=[
                "Enjoy a variety of foods as part of everyday meals",
                "Choose comfortable movement you enjoy",
                "Make room for rest and supportive connections",
                "Explore community activities with people you trust",
            ],
            quote="Wellness is not a destination, but a continuous journey of positive daily choices.",
            org_name="Your Organization Name",
            helpline="Your helpline / website here",
        )

    return PosterCopy(
        headline=preset["headline"],
        slogan=preset["slogan"],
        body=preset["body"],
        bullet_points=list(preset["bullet_points"]),
        quote=preset["quote"],
        org_name=preset["org_name"],
        helpline=preset["helpline"],
    )


# ------------------------------------------------------------------
# Prompt Suggestions
# ------------------------------------------------------------------
def get_example_prompt(category: Optional[str] = None) -> str:
    if category and category in config.CAMPAIGN_PRESETS:
        return config.CAMPAIGN_PRESETS[category]["artwork_prompt"]
    return "Silhouette profile of a human head with a vibrant colorful tree of brain leaves growing inside, promoting mental wellness."


def get_random_suggestion(category: Optional[str] = None) -> str:
    suggestions = {
        "Mental Health": "Silhouette profile of a human head with a vibrant colorful tree of brain leaves growing inside, modern vector art.",
        "Heart Health": "Stylized anatomical lungs and heart vector illustration, clean bold red and deep blue tones on white background.",
        "Blood Donation Awareness": "Modern minimalist geometric blood droplet emblem with vibrant red gradient lines, sleek healthcare logo.",
        "World Health Day": "Diverse people exercising and doing yoga around a clean stylized earth globe, flat modern vector illustration.",
        "Diabetes Awareness": "A colorful balanced bowl of fresh vegetables and whole grains with subtle blue circle health motif.",
        "Healthy Eating": "A vibrant fresh salad bowl overflowing with colorful avocado, tomatoes, greens and citrus, clean vector art.",
        "Physical Fitness": "Energetic silhouettes of diverse people running and stretching outdoors at sunrise, athletic energy.",
        "Women's Health Awareness": "Artistic flat vector illustration of diverse women smiling together with floral botanical accents.",
        "Cancer Awareness": "Elegant awareness ribbon intertwined with hopeful blooming botanical flowers, vibrant soft colors.",
        "General Wellness": "Minimalist clean illustration of a healthy human figure stretching peacefully with green leaves and sunlight rays.",
    }
    if category and category in suggestions:
        return suggestions[category]
    return random.choice(list(suggestions.values()))


# ------------------------------------------------------------------
# Input validation
# ------------------------------------------------------------------
def _find_terms(text: str, terms: List[str]) -> List[str]:
    found = []
    lowered = text.lower()
    for term in terms:
        pattern = r"(?<!\w)" + re.escape(term).replace(r"\ ", r"\s+") + r"(?!\w)"
        if re.search(pattern, lowered):
            found.append(term)
    if re.search(r"\b[A-Z][a-zA-Z-]{2,}\s+\d+\s?(?:mg|mcg|ml)\b", text, re.IGNORECASE):
        found.append("drug name with dosage")
    elif re.search(r"\b\d+\s?(?:mg|mcg|ml)\b", text, re.IGNORECASE):
        found.append("dosage")
    return found


def validate_user_input(text: str, custom_category: str = "",
                        poster_text: str = "") -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    cleaned = (text or "").strip()

    if not cleaned:
        errors.append("Please enter a short description of the illustration or poster theme.")
        return ValidationResult(False, errors, warnings)

    if len(cleaned) < config.MIN_PROMPT_CHARS:
        errors.append(f"Description should be at least {config.MIN_PROMPT_CHARS} characters.")

    if len(cleaned) > config.MAX_PROMPT_CHARS:
        errors.append(f"Description is too long ({len(cleaned)} chars). Max is {config.MAX_PROMPT_CHARS}.")

    visual_input = f"{cleaned} {custom_category or ''}"
    copy_input = poster_text or ""
    comparative_terms = config.COMPARATIVE_IMAGE_TERMS
    blocked_visual = _find_terms(visual_input, config.BLOCKED_TERMS)
    blocked_copy = _find_terms(
        copy_input,
        [term for term in config.BLOCKED_TERMS if term not in comparative_terms],
    )
    comparative = sorted(set(blocked_visual).intersection(comparative_terms))
    other_blocked = sorted(set(blocked_visual + blocked_copy).difference(comparative_terms))
    if comparative:
        errors.append(
            "Before/after comparison imagery isn't supported. Describe one neutral, non-clinical scene instead."
        )
    if other_blocked:
        errors.append(
            "WellAware AI creates non-clinical wellness visuals only. "
            f"Remove the clinical, unsafe, or branded content: **{', '.join(other_blocked)}**."
        )

    combined = f"{visual_input} {copy_input}"
    claims = _find_terms(combined, config.CLAIM_TERMS)
    if claims:
        warnings.append(
            f"Note: Description contains claim-like terms (**{', '.join(sorted(set(claims)))}**). "
            "Please review and revise the poster copy before export."
        )

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)


# ------------------------------------------------------------------
# Prompt building for clean vector artwork on white background
# ------------------------------------------------------------------
def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    # Strip text/word instructions from image prompt since text is rendered separately
    text = re.sub(r"(with|showing|saying)?\s*(the\s+)?(text|caption|slogan|words?)\s*[\"'“][^\"'”]*[\"'”]",
                  "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(please\s+)?(create|generate|make|draw|design)\s+(me\s+)?(an?\s+)?",
                  "", text, flags=re.IGNORECASE)
    return text.strip(" ,.")


def _limit_prompt_words(text: str, limit: int) -> str:
    """Keep the prompt's key phrases near the front of the model's text limit."""
    words = text.split()
    return " ".join(words[:limit])


def build_prompt(user_text: str, category: str, style: str,
                 custom_category: str = "", reference_concepts: str = "",
                 tokenizer=None, audience: str = "") -> PromptBundle:
    """
    Build concise style-aware prompts for text-free campaign artwork.
    """
    description = sanitize_positive_prompt(_clean_text(user_text))

    if category == config.CUSTOM_CATEGORY:
        cat_name = sanitize_positive_prompt((custom_category or "General Wellness").strip())
        cat_visuals = "healthy balanced lifestyle, simple uncluttered composition"
        if not description:
            description = f"clean modern illustration representing {cat_name.lower()}"
    else:
        cat_name = sanitize_positive_prompt(category)
        info = config.CATEGORIES.get(category, config.CATEGORIES["General Wellness"])
        cat_visuals = sanitize_positive_prompt(info["visuals"])
        if not description:
            preset = config.CAMPAIGN_PRESETS.get(category, {})
            description = sanitize_positive_prompt(preset.get("artwork_prompt", cat_visuals))

    # Detect if prompt requests human subjects (woman, doctor, child, patient, senior, etc.)
    has_human = any(
        re.search(r"\b" + re.escape(kw) + r"\b", description.lower())
        for kw in HUMAN_KEYWORDS
    )
    category_has_human = any(
        re.search(r"\b" + re.escape(kw) + r"\b", cat_visuals.lower())
        for kw in HUMAN_KEYWORDS
    )
    contains_people = has_human or category_has_human

    effective_style = style if style in config.STYLES else config.DEFAULT_STYLE
    style_info = config.STYLES.get(effective_style, config.STYLES[config.DEFAULT_STYLE])

    subject = description

    parts = [_limit_prompt_words(description, 32)]
    if cat_visuals and not has_human:
        parts.append(_limit_prompt_words(cat_visuals, 8))
    if effective_style == "Photographic & Realistic":
        style_description = "photorealistic editorial portrait, sharp focus, natural skin texture, soft daylight" if contains_people else "photorealistic editorial lifestyle photograph, sharp focus, natural lighting"
        parts.append(style_description)
    elif style_info.get("prompt"):
        parts.append(_limit_prompt_words(style_info["prompt"], 12))
    parts.append(config.QUALITY_TAGS)
    audience_text = sanitize_positive_prompt(audience)
    prioritized_style = ", ".join(
        phrase for phrase in (
            (
                "one or two people, medium close-up, waist-up portrait, face clearly visible, "
                "looking at camera, natural skin, symmetrical eyes, soft daylight, "
                "simple uncluttered background"
            ) if contains_people else "",
            f"welcoming visual tone for {audience_text}" if audience_text else "",
            style_info.get("prompt", ""),
        ) if phrase
    )
    positive = truncate_prompt_by_priority(
        subject or cat_visuals or cat_name,
        prioritized_style,
        info.get("palette", "") if category != config.CUSTOM_CATEGORY else "",
        ", ".join(parts[1:] + ([reference_concepts] if reference_concepts else [])),
        tokenizer=tokenizer,
    )

    neg_parts = [config.BASE_NEGATIVE_PROMPT]
    if style_info.get("negative"):
        neg_parts.append(_limit_prompt_words(style_info["negative"], 10))
    if has_human:
        neg_parts.append("poorly drawn face, crossed eyes, fused fingers, malformed hands")
    negative = truncate_negative_prompt(", ".join(neg_parts), tokenizer=tokenizer)

    readable = (
        f"A clear {cat_name} visual: {description}. "
        f"Rendered in {effective_style.lower()} style with a simple background."
        + (f" Intended for {audience_text}." if audience_text else "")
    )

    return PromptBundle(
        positive=positive,
        negative=negative,
        category=cat_name,
        style=effective_style,
        user_text=user_text.strip(),
        readable=readable,
    )
