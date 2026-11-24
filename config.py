"""
config.py
---------
Central configuration for WellAware AI.

Every tunable value (model choices, categories, styles, resolutions,
colours, poster layouts, presets, safety lists) lives here.
"""

from pathlib import Path
import os

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"
OUTPUT_DIR = BASE_DIR / "generated_images"
PRESET_POSTER_DIR = OUTPUT_DIR / "preset_campaigns"

# ------------------------------------------------------------------
# App identity
# ------------------------------------------------------------------
APP_NAME = "WellAware AI"
APP_SUBTITLE = "Patient Wellness Campaign Visual Generator"
APP_TEAM = "Team 17"
APP_ICON = "🩺"

# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------
MODELS = {
    "SDXL (GPU quality)": {
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "vae_repo_id": "madebyollin/sdxl-vae-fp16-fix",
        "default_steps": 30,
        "max_steps": 40,
        "default_guidance": 6.5,
        "uses_negative": True,
        "native_size": (1024, 1024),
        "description": "High-detail GPU model. Uses memory-efficient offload and VAE tiling when available.",
    },
    "Stable Diffusion 1.5 (legacy fast)": {
        "repo_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "default_steps": 25,
        "max_steps": 30,
        "default_guidance": 7.5,
        "uses_negative": True,
        "native_size": (512, 512),
        "description": "Legacy lower-resolution local model.",
    },
    "SD-Turbo (draft preview)": {
        "repo_id": "stabilityai/sd-turbo",
        "default_steps": 2,
        "max_steps": 4,
        "default_guidance": 0.0,
        "uses_negative": True,
        "native_size": (512, 512),
        "description": "Fast, lower-detail draft model. Negative prompt support may vary by pipeline.",
    },
}
DEFAULT_MODEL = "SDXL (GPU quality)"
DEFAULT_ENGINE = "Cloud inference" if os.environ.get("SPACE_ID") else "Local GPU"

# ------------------------------------------------------------------
# Poster layouts available in the Streamlit editor
# ------------------------------------------------------------------
POSTER_LAYOUTS = {
    "Clean Hero": {
        "id": "clean_hero",
        "name": "Clean Hero",
        "icon": "Image",
        "description": "Large artwork above a flexible, readable copy area.",
    },
    "Editorial": {
        "id": "editorial",
        "name": "Editorial",
        "icon": "Text",
        "description": "Balanced artwork and copy with the same overflow-safe layout.",
    },
}
DEFAULT_POSTER_LAYOUT = "Clean Hero"

# ------------------------------------------------------------------
# Resolutions (width, height) – for artwork generation
# ------------------------------------------------------------------
RESOLUTIONS = {
    "Square 1024 × 1024 (SDXL)": (1024, 1024),
    "Portrait 832 × 1216 (SDXL)": (832, 1216),
    "Square 512 × 512 (legacy)": (512, 512),
}
DEFAULT_RESOLUTION = "Portrait 832 × 1216 (SDXL)"

POSTER_WIDTH = 1024
POSTER_HEIGHT = 1536
MAX_VARIATIONS = 4

# ------------------------------------------------------------------
# Campaign categories
# ------------------------------------------------------------------
CUSTOM_CATEGORY = "✏️ Custom Category"

CATEGORIES = {
    "General Wellness": {
        "icon": "🌿",
        "visuals": "minimalist clean illustration of a healthy human figure stretching peacefully with green leaves and sunlight rays",
        "palette": "calming health teal, soft aqua, and golden sunlight tones",
    },
    "Free Eye Checkup & Vision Camp": {
        "icon": "👁️",
        "visuals": "friendly eye exam with one optometrist and one adult, bright optical room, clear focus",
        "palette": "clinical cobalt blue, crisp white, and aqua tones",
    },
    "Dental Hygiene & Oral Health Camp": {
        "icon": "🦷",
        "visuals": "friendly dentist and one adult with a natural smile, bright uncluttered clinic",
        "palette": "fresh mint teal, clean sky blue, and pure white tones",
    },
    "Pediatric Care & Vaccination Camp": {
        "icon": "👶",
        "visuals": "caring pediatrician and one parent with a child, cheerful bright clinic",
        "palette": "cheerful sunny yellow, soft sky blue, and gentle coral tones",
    },
    "Senior Mobility & Orthopedic Camp": {
        "icon": "🚶",
        "visuals": "active older adult walking with a companion on a sunny park path",
        "palette": "warm amber gold, sage green, and earthy navy tones",
    },
    "Dengue & Infectious Disease Prevention": {
        "icon": "🛡️",
        "visuals": "community health workers protecting clean water tanks, mosquito net protection, clean hygienic home environment",
        "palette": "protective warning orange, deep navy blue, and clean teal tones",
    },
    "Mental Health": {
        "icon": "🧠",
        "visuals": "calm person meditating in a peaceful garden with leafy shapes",
        "palette": "deep midnight blue, amber gold, and soft teal tones",
    },
    "Heart Health": {
        "icon": "❤️",
        "visuals": "two adults walking together in a green park, gentle heart-shaped foliage",
        "palette": "radiant cyan, coral red, and crisp white tones",
    },
    "Blood Donation Awareness": {
        "icon": "🩸",
        "visuals": "modern geometric blood droplet emblem with vibrant red and coral gradient lines, gift of life, caring community",
        "palette": "deep slate navy, vivid crimson, and warm blush tones",
    },
    "World Health Day": {
        "icon": "🌍",
        "visuals": "two diverse adults enjoying a gentle outdoor activity in a sunny park",
        "palette": "royal health blue, teal green, and cheerful sun yellow tones",
    },
    "Diabetes Awareness": {
        "icon": "🥗",
        "visuals": "colorful balanced bowl of fresh vegetables and whole grains, active people walking, blue circle wellness motif",
        "palette": "sky blue, fresh green, and clean white tones",
    },
    "Healthy Eating": {
        "icon": "🥑",
        "visuals": "a healthy happy smiling woman holding a fresh bowl of ripe fruits and green vegetables, bright natural sunlight, kitchen daylight",
        "palette": "fresh leafy green, carrot orange, and warm lime tones",
    },
    "Physical Fitness": {
        "icon": "🏃",
        "visuals": "two diverse adults enjoying a relaxed walk outdoors at sunrise",
        "palette": "energetic athletic teal, vibrant orange, and white tones",
    },
    "Women's Health Awareness": {
        "icon": "🌸",
        "visuals": "two diverse women sharing a relaxed conversation in a botanical garden",
        "palette": "soft rose, blossom pink, and coral tones",
    },
    "Cancer Awareness": {
        "icon": "🎗️",
        "visuals": "elegant awareness ribbon intertwined with hopeful blooming botanical flowers, solidarity, hope, early screening",
        "palette": "awareness purple, hope pink, and gentle lavender tones",
    },
}

# ------------------------------------------------------------------
# Visual styles
# ------------------------------------------------------------------
STYLES = {
    "Flat Vector Graphic": {
        "icon": "📐",
        "prompt": "modern flat vector illustration, clean geometric shapes, expressive forms, harmonious colors",
        "negative": "blurry, grainy, 3d render, noisy texture, photorealistic",
    },
    "Photographic & Realistic": {
        "icon": "📷",
        "prompt": "clean professional editorial lifestyle photography, crisp subject focus, natural soft lighting, high resolution 35mm film aesthetic, clean composition",
        "negative": "cartoon, anime, 3d render, illustration, sketch, blurry, oversaturated, deformed",
    },
    "Modern 3D Digital & Clay": {
        "icon": "🧊",
        "prompt": "3D digital artwork, smooth clay style, soft studio rim lighting, vibrant depth, tactile materials, modern graphic render",
        "negative": "flat 2d, sketch, rough drawing, blurry, grainy",
    },
    "Minimalist Modern Geometry": {
        "icon": "⚪",
        "prompt": "minimalist graphic design, bold iconic silhouettes, elegant negative space, clean geometric composition",
        "negative": "cluttered, messy, noisy, low resolution, blurry",
    },
    "Watercolor & Fine Art": {
        "icon": "🖌️",
        "prompt": "soft expressive watercolor painting, luminous color washes, delicate ink line art, artistic and hopeful, clean paper texture",
        "negative": "harsh plastic 3d render, photo, neon, dark muddy colors",
    },
    "Bold Pop & Screenprint": {
        "icon": "🎨",
        "prompt": "bold contemporary screenprint poster art, high contrast, vibrant graphic color blocking, striking editorial visual",
        "negative": "blurry, dull colors, low resolution, messy artifacts",
    },
}
DEFAULT_STYLE = "Photographic & Realistic"

# ------------------------------------------------------------------
# Rich Campaign Presets (Matching Real Healthcare Campaigns)
# ------------------------------------------------------------------
CAMPAIGN_PRESETS = {
    "Mental Health": {
        "headline": "MENTAL HEALTH MONTH",
        "slogan": "Take Care of What's Inside",
        "body": "This month, let's break the stigma and start the conversation. Mental health is just as important as physical health. Take time to rest, connect, and care for your mind.",
        "bullet_points": [
            "Talk openly about mental health with loved ones",
            "Reach out for support whenever you feel overwhelmed",
            "Practice daily self-care, healthy boundaries & balance",
            "Support and listen to others with kindness & patience",
        ],
        "quote": "Together, we can build a world where mental health is prioritized, understood, and supported.",
        "org_name": "Your Organization Name",
        "helpline": "Your helpline / website here",
        "artwork_prompt": "one person relaxing in a peaceful garden with colorful leafy shapes",
    },
    "Heart Health": {
        "headline": "HEART HEALTH AWARENESS",
        "slogan": "Strong Heart, Healthy Lungs, Full Life",
        "body": "Make room for everyday movement, restful moments, nourishing meals, and time with people you care about.",
        "bullet_points": [
            "Choose movement that feels comfortable and enjoyable",
            "Add a variety of colorful foods to everyday meals",
            "Make time for rest and supportive connections",
            "Ask a qualified professional about personal wellness questions",
        ],
        "quote": "Small, thoughtful choices can bring more balance to everyday life.",
        "org_name": "Your Organization Name",
        "helpline": "Your helpline / website here",
        "artwork_prompt": "two adults walking together in a green park, gentle heart-shaped foliage, warm red and deep blue colors",
    },
    "Blood Donation Awareness": {
        "headline": "COMMUNITY CARE TOGETHER",
        "slogan": "A moment of care can bring people together",
        "body": "Learn about community wellness activities and make time for supportive conversations with people around you.",
        "bullet_points": [
            "Explore local community wellness activities",
            "Make time for rest, hydration, and a balanced routine",
            "Share encouragement with friends and neighbors",
            "Seek guidance from qualified local professionals",
        ],
        "quote": "Kindness and community support make a meaningful difference.",
        "org_name": "Your Organization Name",
        "helpline": "Your helpline / website here",
        "artwork_prompt": "a caring community sharing a quiet moment together, warm red and coral colors",
    },
    "World Health Day": {
        "headline": "WORLD HEALTH DAY",
        "slogan": "My Health, My Right · Wellness for Every Community",
        "body": "Wellness can include enjoyable movement, varied meals, restful moments, and supportive relationships.",
        "bullet_points": [
            "Find a form of movement that fits your day",
            "Enjoy a variety of foods and drink water regularly",
            "Make time for rest and positive social connections",
            "Explore community activities with family or friends",
        ],
        "quote": "Wellness grows through small moments of care and connection.",
        "org_name": "Your Organization Name",
        "helpline": "Your helpline / website here",
        "artwork_prompt": "two diverse adults enjoying a relaxed outdoor activity together in a sunny park",
    },
    "Diabetes Awareness": {
        "headline": "DIABETES AWARENESS",
        "slogan": "Empower Your Life Through Balanced Living",
        "body": "Explore balanced routines that include varied meals, enjoyable movement, rest, and supportive conversations.",
        "bullet_points": [
            "Enjoy a variety of foods as part of your everyday routine",
            "Choose comfortable movement you can enjoy",
            "Make space for rest and supportive connections",
            "For personal questions, speak with a qualified professional",
        ],
        "quote": "Small, consistent daily habits build the strongest defense for lifelong vitality.",
        "org_name": "Your Organization Name",
        "helpline": "Your helpline / website here",
        "artwork_prompt": "colorful bowl of vegetables and grains on a bright kitchen table, gentle blue accents",
    },
    "Healthy Eating": {
        "headline": "EAT FRESH, LIVE VIBRANT",
        "slogan": "Nourish Your Body from the Inside Out",
        "body": "Make everyday meals colorful with a variety of fruits, vegetables, grains, and other familiar foods.",
        "bullet_points": [
            "Try a variety of colors and textures in meals",
            "Keep water nearby throughout the day",
            "Share meal preparation with family or friends",
            "Enjoy meals at a comfortable pace",
        ],
        "quote": "A colorful meal can be a joyful part of the day.",
        "org_name": "Your Organization Name",
        "helpline": "Your helpline / website here",
        "artwork_prompt": "a vibrant bowl of ripe fruits, leafy vegetables, tomatoes, and citrus in a sunlit kitchen",
    },
    "Physical Fitness": {
        "headline": "EVERYDAY MOVEMENT",
        "slogan": "Find a way to move that feels right for you",
        "body": "Movement can be part of an enjoyable day, from a relaxed walk to stretching with a friend.",
        "bullet_points": [
            "Choose activities that feel comfortable and enjoyable",
            "Try gentle stretching when it fits your routine",
            "Take a movement break during a long seated activity",
            "Invite a friend to join you for a walk",
        ],
        "quote": "You don't have to be extreme, just consistent. Every step moves you forward.",
        "org_name": "Your Organization Name",
        "helpline": "Your helpline / website here",
        "artwork_prompt": "two diverse adults walking together in a sunny park at sunrise",
    },
    "Women's Health Awareness": {
        "headline": "WOMEN'S WELLNESS FIRST",
        "slogan": "Nurturing Strength, Harmony, and Lifelong Care",
        "body": "Make space for rest, movement, nourishing meals, and supportive relationships at every stage of life.",
        "bullet_points": [
            "Choose movement that feels comfortable and enjoyable",
            "Make time for rest and supportive conversations",
            "Enjoy a variety of foods as part of everyday meals",
            "Explore community activities with people you trust",
        ],
        "quote": "Care, confidence, and connection belong in every season of life.",
        "org_name": "Your Organization Name",
        "helpline": "Your helpline / website here",
        "artwork_prompt": "two diverse women sharing a relaxed conversation in a bright botanical garden",
    },
    "Cancer Awareness": {
        "headline": "HOPE, CARE & CONNECTION",
        "slogan": "A moment of care can bring people together",
        "body": "Share encouragement, make time for meaningful conversations, and explore local community resources.",
        "bullet_points": [
            "Make time for supportive conversations",
            "Explore community activities and local resources",
            "Choose moments of rest and enjoyable movement",
            "Offer kindness and encouragement to people around you",
        ],
        "quote": "Hope is the anchor of the soul and the spark that ignites our collective strength.",
        "org_name": "Your Organization Name",
        "helpline": "Your helpline / website here",
        "artwork_prompt": "a purple ribbon among blooming flowers in a peaceful garden, soft hopeful colors",
    },
    "General Wellness": {
        "headline": "WELLNESS STARTS TODAY",
        "slogan": "Small Healthy Steps for a Lifetime of Vitality",
        "body": "Wellness can include enjoyable movement, varied meals, restful moments, and supportive relationships.",
        "bullet_points": [
            "Keep water nearby and drink when you feel thirsty",
            "Make room for restful moments in your day",
            "Try a calming activity that you enjoy",
            "Spend time outdoors when it suits you",
        ],
        "quote": "Small moments of care can make everyday life feel more balanced.",
        "org_name": "Your Organization Name",
        "helpline": "Your helpline / website here",
        "artwork_prompt": "one person stretching peacefully in a sunlit garden with green leaves",
    },
    "Free Eye Checkup & Vision Camp": {
        "headline": "CLEAR VISION FOR LIFE",
        "slogan": "Protect Your Eyesight, Brighten Your World",
        "body": "Create a comfortable space for reading, screen breaks, and enjoying the world around you.",
        "bullet_points": [
            "Take a pause from screens when convenient",
            "Use comfortable lighting for reading",
            "Enjoy time outdoors in a way that feels comfortable",
            "Ask a qualified professional about personal eye-care questions",
        ],
        "quote": "The eyes are the windows to the soul and the guide to a vibrant life.",
        "org_name": "Your Organization Name",
        "helpline": "Your helpline / website here",
        "artwork_prompt": "an optometrist and one adult sharing a friendly conversation in a bright clinic",
    },
    "Dental Hygiene & Oral Health Camp": {
        "headline": "HEALTHY SMILES, HEALTHY LIFE",
        "slogan": "Make room for everyday oral-care routines",
        "body": "A relaxed daily routine can include brushing, caring for your smile, and asking questions when you need guidance.",
        "bullet_points": [
            "Build a routine that works for your day",
            "Keep everyday oral-care items in a convenient place",
            "Make time for regular breaks and water",
            "Ask a qualified dental professional for personal guidance",
        ],
        "quote": "A genuine smile is the shortest distance between two people.",
        "org_name": "Your Organization Name",
        "helpline": "Your helpline / website here",
        "artwork_prompt": "a friendly dentist and one adult with natural smiles in a bright uncluttered clinic",
    },
    "Pediatric Care & Vaccination Camp": {
        "headline": "NURTURING CHILD WELLNESS",
        "slogan": "Healthy Steps for Little Champions",
        "body": "Make time for play, rest, nourishing meals, and supportive moments together.",
        "bullet_points": [
            "Make space for joyful play and creative activities",
            "Enjoy a variety of foods together",
            "Build in quiet moments for rest",
            "Encourage kind conversations and connection",
        ],
        "quote": "Children are our greatest treasure and the future of our communities.",
        "org_name": "Your Organization Name",
        "helpline": "Your helpline / website here",
        "artwork_prompt": "a pediatrician and one child sharing a warm moment in a bright clinic",
    },
    "Senior Mobility & Orthopedic Camp": {
        "headline": "ACTIVE & HEALTHY AGING",
        "slogan": "Vitality, Wisdom, and Grace at Every Stage of Life",
        "body": "Everyday routines can include comfortable movement, meaningful conversations, and moments of rest.",
        "bullet_points": [
            "Choose comfortable movement that suits you",
            "Stay connected with friends, family, or community",
            "Enjoy a variety of foods and restful moments",
            "Ask a qualified professional about personal wellness needs",
        ],
        "quote": "Age is an honor and a testament to a life lived with purpose and care.",
        "org_name": "Your Organization Name",
        "helpline": "Your helpline / website here",
        "artwork_prompt": "an older adult walking with a companion on a quiet sunny park path",
    },
    "Dengue & Infectious Disease Prevention": {
        "headline": "COMMUNITY WELLNESS",
        "slogan": "Clean Surroundings, Safe and Healthy Communities",
        "body": "A clean, comfortable shared space can support everyday wellbeing and community connection.",
        "bullet_points": [
            "Keep shared spaces tidy and comfortable",
            "Spend time together on simple neighborhood activities",
            "Make room for rest and enjoyable outdoor time",
            "Ask qualified local professionals for personal guidance",
        ],
        "quote": "A clean community is the first and strongest barrier against disease.",
        "org_name": "Your Organization Name",
        "helpline": "Your helpline / website here",
        "artwork_prompt": "one community member tending a sunny garden beside a clean water container",
    },
}

# ------------------------------------------------------------------
# Poster Color Palettes (Rich Healthcare Palettes)
# ------------------------------------------------------------------
POSTER_PALETTES = {
    "Mental Health": {
        "primary": "#0D253F",       # Deep navy (like Ref 3)
        "accent": "#F5A623",        # Warm gold / amber
        "accent2": "#2EB872",       # Soft teal/green
        "bg_light": "#FFFDF8",      # Warm cream
        "bg_dark": "#0B1D30",       # Dark card
        "text_dark": "#0D253F",
        "text_light": "#FFFFFF",
        "badge_bg": "#F5A623",
    },
    "Heart Health": {
        "primary": "#0B7A8B",       # Cyan/Teal (like Ref 1 sunburst)
        "accent": "#E63946",        # Red / coral
        "accent2": "#457B9D",       # Steel blue
        "bg_light": "#F0F9FB",      # Light cyan tint
        "bg_dark": "#0A3641",
        "text_dark": "#0B3039",
        "text_light": "#FFFFFF",
        "badge_bg": "#E63946",
    },
    "Blood Donation Awareness": {
        "primary": "#1A365D",       # Deep slate navy (like Ref 4)
        "accent": "#E63946",        # Vivid blood red
        "accent2": "#F4A261",       # Coral pink gradient
        "bg_light": "#FFF5F2",      # Warm blush cream
        "bg_dark": "#132338",
        "text_dark": "#1A365D",
        "text_light": "#FFFFFF",
        "badge_bg": "#E63946",
    },
    "World Health Day": {
        "primary": "#1E3A8A",       # Royal health blue (like Ref 2)
        "accent": "#0D9488",        # Teal green
        "accent2": "#F59E0B",       # Sun yellow
        "bg_light": "#FFFFFF",      # Crisp clean white
        "bg_dark": "#111827",
        "text_dark": "#111827",
        "text_light": "#FFFFFF",
        "badge_bg": "#0D9488",
    },
    "Diabetes Awareness": {
        "primary": "#0369A1",       # Blue
        "accent": "#06B6D4",        # Sky cyan
        "accent2": "#10B981",       # Emerald
        "bg_light": "#F0F9FF",
        "bg_dark": "#082F49",
        "text_dark": "#082F49",
        "text_light": "#FFFFFF",
        "badge_bg": "#0284C7",
    },
    "Healthy Eating": {
        "primary": "#166534",       # Deep garden green
        "accent": "#84CC16",        # Fresh lime
        "accent2": "#F97316",       # Vibrant carrot orange
        "bg_light": "#F7FEE7",
        "bg_dark": "#14532D",
        "text_dark": "#14532D",
        "text_light": "#FFFFFF",
        "badge_bg": "#15803D",
    },
    "Physical Fitness": {
        "primary": "#0F766E",       # Deep athletic teal
        "accent": "#F97316",        # High energy orange
        "accent2": "#06B6D4",       # Sky blue
        "bg_light": "#F0FDFA",
        "bg_dark": "#134E4A",
        "text_dark": "#134E4A",
        "text_light": "#FFFFFF",
        "badge_bg": "#F97316",
    },
    "Women's Health Awareness": {
        "primary": "#9D174D",       # Rich rose / berry
        "accent": "#F472B6",        # Blossom pink
        "accent2": "#FB7185",       # Coral
        "bg_light": "#FDF2F8",
        "bg_dark": "#831843",
        "text_dark": "#831843",
        "text_light": "#FFFFFF",
        "badge_bg": "#DB2777",
    },
    "Cancer Awareness": {
        "primary": "#701A75",       # Awareness purple / plum
        "accent": "#F43F5E",        # Hope pink
        "accent2": "#C084FC",       # Lavender
        "bg_light": "#FAF5FF",
        "bg_dark": "#581C87",
        "text_dark": "#581C87",
        "text_light": "#FFFFFF",
        "badge_bg": "#9333EA",
    },
    "General Wellness": {
        "primary": "#0F766E",       # Health teal
        "accent": "#14B8A6",        # Aqua
        "accent2": "#F59E0B",       # Golden sunlight
        "bg_light": "#F0FDFA",
        "bg_dark": "#115E59",
        "text_dark": "#134E4A",
        "text_light": "#FFFFFF",
        "badge_bg": "#0D9488",
    },
    "Free Eye Checkup & Vision Camp": {
        "primary": "#0369A1",
        "accent": "#06B6D4",
        "accent2": "#38BDF8",
        "bg_light": "#F0F9FF",
        "bg_dark": "#082F49",
        "text_dark": "#082F49",
        "text_light": "#FFFFFF",
        "badge_bg": "#0284C7",
    },
    "Dental Hygiene & Oral Health Camp": {
        "primary": "#0D9488",
        "accent": "#14B8A6",
        "accent2": "#2DD4BF",
        "bg_light": "#F0FDFA",
        "bg_dark": "#115E59",
        "text_dark": "#115E59",
        "text_light": "#FFFFFF",
        "badge_bg": "#0D9488",
    },
    "Pediatric Care & Vaccination Camp": {
        "primary": "#D97706",
        "accent": "#F59E0B",
        "accent2": "#FB7185",
        "bg_light": "#FFFBEB",
        "bg_dark": "#78350F",
        "text_dark": "#78350F",
        "text_light": "#FFFFFF",
        "badge_bg": "#D97706",
    },
    "Senior Mobility & Orthopedic Camp": {
        "primary": "#475569",
        "accent": "#F59E0B",
        "accent2": "#10B981",
        "bg_light": "#F8FAFC",
        "bg_dark": "#1E293B",
        "text_dark": "#1E293B",
        "text_light": "#FFFFFF",
        "badge_bg": "#475569",
    },
    "Dengue & Infectious Disease Prevention": {
        "primary": "#EA580C",
        "accent": "#F97316",
        "accent2": "#0D9488",
        "bg_light": "#FFF7ED",
        "bg_dark": "#7C2D12",
        "text_dark": "#7C2D12",
        "text_light": "#FFFFFF",
        "badge_bg": "#EA580C",
    },
}

THEME_PALETTES = {
    "✨ Auto-Detect from Prompt & Campaign": None,
    "💜 Awareness Purple & Golden Hope": {
        "primary": "#701A75", "accent": "#F5A623", "accent2": "#C084FC",
        "bg_light": "#FAF5FF", "bg_dark": "#3B0764", "text_dark": "#3B0764", "text_light": "#FFFFFF", "badge_bg": "#9333EA",
    },
    "💙 Royal Health Blue & Teal": {
        "primary": "#1E3A8A", "accent": "#0D9488", "accent2": "#38BDF8",
        "bg_light": "#F0F9FF", "bg_dark": "#0B1D30", "text_dark": "#0B1D30", "text_light": "#FFFFFF", "badge_bg": "#0D9488",
    },
    "💚 Fresh Botanical Green & Lime": {
        "primary": "#166534", "accent": "#84CC16", "accent2": "#F59E0B",
        "bg_light": "#F7FEE7", "bg_dark": "#14532D", "text_dark": "#14532D", "text_light": "#FFFFFF", "badge_bg": "#15803D",
    },
    "❤️ Crimson Blood Red & Slate Navy": {
        "primary": "#1A365D", "accent": "#E63946", "accent2": "#F4A261",
        "bg_light": "#FFF5F2", "bg_dark": "#132338", "text_dark": "#132338", "text_light": "#FFFFFF", "badge_bg": "#E63946",
    },
    "🧡 Athletic Sunrise Orange & Cyan": {
        "primary": "#0F766E", "accent": "#F97316", "accent2": "#06B6D4",
        "bg_light": "#F0FDFA", "bg_dark": "#134E4A", "text_dark": "#134E4A", "text_light": "#FFFFFF", "badge_bg": "#F97316",
    },
    "🌸 Rose Blossom & Warm Coral": {
        "primary": "#9D174D", "accent": "#F472B6", "accent2": "#FB7185",
        "bg_light": "#FDF2F8", "bg_dark": "#831843", "text_dark": "#831843", "text_light": "#FFFFFF", "badge_bg": "#DB2777",
    },
    "🌙 Modern Midnight Dark & Cyan Glow": {
        "primary": "#0284C7", "accent": "#38BDF8", "accent2": "#22D3EE",
        "bg_light": "#F8FAFC", "bg_dark": "#0F172A", "text_dark": "#0F172A", "text_light": "#FFFFFF", "badge_bg": "#0284C7",
    },
}

# ------------------------------------------------------------------
# Prompt engineering constants
# ------------------------------------------------------------------
QUALITY_TAGS = "clean editorial lifestyle image, balanced composition, vivid harmonious colors, sharp focus, natural detail"

BASE_NEGATIVE_PROMPT = (
    "text, letters, words, numbers, caption, signature, watermark, logo, sign, label, banner, "
    "gibberish, handwriting, qr code, ui, border, frame, blurry, out of focus, low resolution, "
    "jpeg artifacts, deformed face, asymmetrical eyes, unnatural smile, distorted anatomy, "
    "malformed hands, extra fingers, missing fingers, fused fingers, malformed limbs, duplicate people"
)

# ------------------------------------------------------------------
# Input validation & safety
# ------------------------------------------------------------------
MIN_PROMPT_CHARS = 3
MAX_PROMPT_CHARS = 500
MAX_HEADLINE_CHARS = 180
MAX_SLOGAN_CHARS = 300
MAX_BODY_CHARS = 2000
MAX_TIPS_CHARS = 2000
MAX_QUOTE_CHARS = 800
MAX_ORG_NAME_CHARS = 160
MAX_HELPLINE_CHARS = 300

BLOCKED_TERMS = [
    # clinical / diagnostic imagery
    "x-ray", "xray", "mri", "ct scan", "ultrasound", "biopsy", "tumor", "tumour",
    "surgery", "surgical", "operation theatre", "autopsy", "diagnosis", "diagnose",
    "diagnostic", "lab report", "prescription", "dosage",
    # graphic / unsafe content
    "blood and gore", "bloody wound", "violent blood", "excessive bleeding", "blood splatter",
    "gore", "wound", "corpse", "dead body", "suicide", "self-harm", "self harm",
    "nude", "nudity", "naked", "nsfw", "sexual", "weapon", "gun", "violence",
    # pharma branding
    "pfizer", "novartis", "johnson & johnson", "astrazeneca", "moderna", "roche",
    "merck", "sanofi", "gsk", "bayer", "abbott", "eli lilly", "lipitor", "ozempic",
    "before and after", "before/after", "before-after",
]
COMPARATIVE_IMAGE_TERMS = ["before and after", "before/after", "before-after"]

CLAIM_TERMS = [
    "cure", "cures", "curing", "guaranteed", "guarantee", "100%", "miracle", "heals",
    "treats", "treatment", "saves lives", "save lives", "prevents cancer", "prevents infections",
    "prevents disease", "clinically proven", "reverses diabetes", "reversing diabetes",
]

DISCLAIMER_SHORT = (
    "AI-generated, non-clinical wellness imagery. Requires internal medical/legal/regulatory "
    "(MLR) review before public use."
)

DISCLAIMER_FULL = (
    "WellAware AI creates **illustrative, non-clinical** wellness and awareness visuals "
    "using a pretrained AI model. Generated images are **not medical advice**, may contain "
    "inaccuracies, and **must be reviewed by a qualified person** (medical, legal, brand) "
    "before publication. Do not use these images to imply medical claims or diagnoses."
)

AUDIT_LOG_PATH = BASE_DIR / "auditlog.csv"
HF_CLOUD_MODEL = "black-forest-labs/FLUX.1-schnell"
HF_CLOUD_FALLBACK_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
CLIP_TOKENIZER_ID = "openai/clip-vit-large-patch14"

# ------------------------------------------------------------------
# UI colours (healthcare palette)
# ------------------------------------------------------------------
COLORS = {
    "primary": "#0B7774",
    "primary_dark": "#075E5B",
    "secondary": "#1769AA",
    "bg_soft": "#F2FAFB",
    "text": "#1F3B4D",
    "muted": "#526B78",
    "border": "#DCEFF2",
}
