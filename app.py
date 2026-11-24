"""Streamlit workflow for the WellAware AI campaign visual generator."""

from __future__ import annotations

import hashlib
import inspect
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

import config
from modules import image_utils
from modules.preset_campaigns import delete_saved_gallery_poster, load_saved_gallery_posters
from modules.image_generator import (
    GenerationError,
    WellnessImageGenerator,
    check_dependencies,
    get_device_info,
    unload_cuda_memory,
)
from modules.prompt_processor import (
    PosterCopy,
    build_prompt,
    count_clip_tokens,
    generate_poster_copy,
    get_campaign_preset,
    get_example_prompt,
    get_random_suggestion,
    lint_poster_copy,
    validate_user_input,
)
load_dotenv(config.BASE_DIR / ".env")

# ==================================================================
# Page setup
# ==================================================================
st.set_page_config(
    page_title=f"{config.APP_NAME} · {config.APP_SUBTITLE}",
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)

LOGO = str(image_utils.ensure_logo())

STRETCH = ({"width": "stretch"} if "width" in inspect.signature(st.button).parameters
           else {"use_container_width": True})
C = config.COLORS
PAGES = ["Home", "Create Campaign", "Gallery", "Guidelines"]


# ==================================================================
# Styling
# ==================================================================
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"], .stMarkdown, button, input, textarea {{
            font-family: 'Plus Jakarta Sans', sans-serif;
        }}
        .stApp {{ background: #FAFCFD; }}
        .block-container {{ padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1280px; }}
        h1, h2, h3, h4 {{ color: {C['text']}; font-weight: 700; }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {C['bg_soft']} 0%, #FFFFFF 100%);
            border-right: 1px solid {C['border']};
        }}
        .sb-brand {{ font-size: 1.35rem; font-weight: 800; color: {C['primary_dark']}; margin: 0; }}
        .sb-sub {{ font-size: .8rem; color: {C['muted']}; margin-top: -4px; }}

        /* Buttons */
        .stButton > button, .stDownloadButton > button {{
            border-radius: 12px; font-weight: 600; border: 1px solid {C['border']};
            transition: all .15s ease-in-out;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            border-color: {C['primary']}; color: {C['primary_dark']};
            transform: translateY(-1px); box-shadow: 0 4px 12px rgba(15,157,154,.15);
        }}
        .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
            background: linear-gradient(90deg, {C['primary']} 0%, {C['secondary']} 100%);
            color: white; border: none;
        }}
        .stButton > button[kind="primary"]:hover {{ color: white; filter: brightness(1.05); }}

        /* Inputs */
        textarea, input, [data-baseweb="select"] > div {{ border-radius: 10px !important; }}

        /* Bordered containers */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            border-radius: 16px !important; border-color: {C['border']} !important;
            background: #FFFFFF;
        }}

        /* Hero */
        .hero {{
            background: linear-gradient(125deg, #0D253F 0%, #0B7774 60%, #1769AA 100%);
            border-radius: 22px; padding: 2.6rem 2.4rem; color: white;
            box-shadow: 0 12px 30px rgba(13,37,63,.22); margin-bottom: 1.6rem;
        }}
        .hero h1 {{ color: white; font-size: 2.6rem; margin: 0; font-weight: 800; }}
        .hero h3 {{ color: #E6FAF9; font-weight: 500; margin: .3rem 0 1rem 0; }}
        .hero p  {{ font-size: 1.05rem; max-width: 820px; opacity: .95; }}
        .badge {{
            display: inline-block; background: rgba(255,255,255,.18); color: white;
            padding: .25rem .8rem; border-radius: 999px; font-size: .8rem; margin-right: .4rem;
        }}
        .card {{
            background: #FFFFFF; border: 1px solid {C['border']}; border-radius: 16px;
            padding: 1.2rem 1.2rem; height: 100%;
            box-shadow: 0 2px 10px rgba(31,59,77,.04);
        }}
        .card .icon {{ font-size: 1.8rem; }}
        .card h4 {{ margin: .4rem 0 .3rem 0; font-size: 1.02rem; }}
        .card p  {{ color: {C['muted']}; font-size: .9rem; margin: 0; }}
        .chip {{
            display: inline-block; background: {C['bg_soft']}; color: {C['primary_dark']};
            border: 1px solid {C['border']}; padding: .2rem .7rem; border-radius: 999px;
            font-size: .8rem; font-weight: 600; margin: 0 .3rem .3rem 0;
        }}
        .section-title {{ font-size: 1.25rem; font-weight: 700; color: {C['text']}; margin: 1.4rem 0 .6rem 0; }}
        .label {{ font-weight: 700; color: {C['text']}; margin-bottom: .25rem; font-size: .92rem; }}
        .muted {{ color: {C['muted']}; font-size: .88rem; }}
        .disclaimer {{
            background: #FFF9EC; border: 1px solid #F6E3B4; border-radius: 14px;
            padding: .9rem 1.1rem; color: #6B5414; font-size: .88rem;
        }}
        .empty {{
            border: 2px dashed {C['border']}; border-radius: 18px; padding: 3rem 1.5rem;
            text-align: center; color: {C['muted']}; background: {C['bg_soft']};
        }}
        .empty .big {{ font-size: 3rem; }}
        .footer {{ text-align: center; color: {C['muted']}; font-size: .8rem; margin-top: 2.5rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==================================================================
# Session state & Callbacks
# ==================================================================
def init_state() -> None:
    first_cat = list(config.CATEGORIES)[0]
    preset = get_campaign_preset(first_cat)
    default_engine = config.DEFAULT_ENGINE
    defaults = {
        "nav": PAGES[0],
        "last_results": [],
        "last_request": None,
        "pending_request": None,
        "user_prompt": "One friendly healthcare professional in blue scrubs, smiling in a bright modern clinic, hands out of frame.",
        "category": first_cat,
        "custom_category": "",
        "poster_layout": config.DEFAULT_POSTER_LAYOUT,
        "poster_title": preset.headline,
        "poster_slogan": preset.slogan,
        "poster_body": preset.body,
        "poster_tips": "\n".join(preset.bullet_points),
        "poster_quote": preset.quote,
        "poster_org": preset.org_name,
        "poster_helpline": preset.helpline,
        "style": config.DEFAULT_STYLE,
        "audience": "General adults",
        "theme_choice": "✨ Auto-Detect from Prompt & Campaign",
        "engine_mode": default_engine,
        "model_label": config.DEFAULT_MODEL,
        "prompt_token_count": None,
        "first_visit": True,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if "gallery" not in st.session_state:
        st.session_state.gallery = []
    if not st.session_state.get("disk_gallery_loaded"):
        try:
            saved_posters = load_saved_gallery_posters()
            existing_ids = {item["id"] for item in st.session_state.gallery}
            existing_paths = {
                str(Path(item["saved_path"]).resolve())
                for item in st.session_state.gallery
                if item.get("saved_path")
            }
            for item in reversed(saved_posters):
                saved_path = str(Path(item["saved_path"]).resolve())
                if item["id"] not in existing_ids and saved_path not in existing_paths:
                    st.session_state.gallery.insert(0, item)
                    existing_ids.add(item["id"])
                    existing_paths.add(saved_path)
            st.session_state.disk_gallery_loaded = True
        except (OSError, ValueError) as exc:
            st.error(f"Could not load saved posters: {exc}")

    previous_preset_posters = st.session_state.pop("preset_posters", {})
    existing_ids = {item["id"] for item in st.session_state.gallery}
    for item in previous_preset_posters.values():
        if item["id"] not in existing_ids:
            st.session_state.gallery.insert(0, item)
            existing_ids.add(item["id"])
    if st.session_state.get("nav") == "Ready-made Posters":
        st.session_state.nav = "Gallery"
    elif st.session_state.get("nav") not in PAGES:
        st.session_state.nav = PAGES[0]

    if st.session_state.get("user_prompt", "").strip() == (
        "A friendly healthcare professional in blue scrubs, smiling in a bright modern clinic, "
        "waist-up portrait, hands out of frame."
    ):
        st.session_state.user_prompt = defaults["user_prompt"]

    if st.session_state.get("engine_mode") != default_engine:
        st.session_state.engine_mode = default_engine
    if st.session_state.get("poster_layout") not in config.POSTER_LAYOUTS:
        st.session_state.poster_layout = config.DEFAULT_POSTER_LAYOUT

    if not st.session_state.get("_quality_defaults_migrated"):
        if st.session_state.get("model_label") not in config.MODELS:
            st.session_state.model_label = config.DEFAULT_MODEL
            if st.session_state.get("style") == "Flat Vector Graphic":
                st.session_state.style = config.DEFAULT_STYLE
        st.session_state["_quality_defaults_migrated"] = True


def go_to(page: str) -> None:
    st.session_state.nav = page


def get_recommended_layout(category: str) -> str:
    return "Editorial" if "Mental" in category else config.DEFAULT_POSTER_LAYOUT


def cb_category_change() -> None:
    """Triggered automatically when the user selects a new campaign category."""
    cat = st.session_state.category
    if cat != config.CUSTOM_CATEGORY and cat in config.CAMPAIGN_PRESETS:
        preset = config.CAMPAIGN_PRESETS[cat]
        # Only fill prompt if user hasn't typed an expressive custom prompt
        current = st.session_state.get("user_prompt", "").strip()
        is_preset = any(current == p.get("artwork_prompt", "").strip() for p in config.CAMPAIGN_PRESETS.values())
        if not current or is_preset:
            st.session_state.user_prompt = preset.get("artwork_prompt", "")
        st.session_state.poster_title = preset["headline"]
        st.session_state.poster_slogan = preset["slogan"]
        st.session_state.poster_body = preset["body"]
        st.session_state.poster_tips = "\n".join(preset["bullet_points"])
        st.session_state.poster_quote = preset["quote"]
        st.session_state.poster_org = preset["org_name"]
        st.session_state.poster_helpline = preset["helpline"]
        st.session_state.poster_layout = get_recommended_layout(cat)
    elif cat == config.CUSTOM_CATEGORY:
        st.session_state.poster_layout = config.DEFAULT_POSTER_LAYOUT


def cb_autofill_copy() -> None:
    cat = st.session_state.get("category", "Mental Health")
    cust = st.session_state.get("custom_category", "")
    prompt = st.session_state.get("user_prompt", "")
    copy = generate_poster_copy(prompt, cat, cust)

    st.session_state.poster_title = copy.headline
    st.session_state.poster_slogan = copy.slogan
    st.session_state.poster_body = copy.body
    st.session_state.poster_tips = "\n".join(copy.bullet_points)
    st.session_state.poster_quote = copy.quote
    st.session_state.poster_org = copy.org_name
    st.session_state.poster_helpline = copy.helpline
    st.session_state.poster_layout = get_recommended_layout(cat)


def cb_example_prompt() -> None:
    cat = st.session_state.get("category", "Mental Health")
    st.session_state.user_prompt = get_example_prompt(cat)
    cb_autofill_copy()


def cb_random_prompt() -> None:
    cat = st.session_state.get("category")
    st.session_state.user_prompt = get_random_suggestion(cat if cat in config.CATEGORIES else None)
    cb_autofill_copy()


def cb_quick_prompt(prompt_text: str, category_override: Optional[str] = None, theme_override: Optional[str] = None) -> None:
    st.session_state.user_prompt = prompt_text
    if category_override and category_override in config.CATEGORIES:
        st.session_state.category = category_override
        st.session_state.poster_layout = get_recommended_layout(category_override)
    if theme_override and theme_override in getattr(config, "THEME_PALETTES", {}):
        st.session_state.theme_choice = theme_override
    cb_autofill_copy()


def cb_clear_prompt() -> None:
    st.session_state.user_prompt = ""
    st.session_state.poster_title = ""
    st.session_state.poster_slogan = ""
    st.session_state.poster_body = ""
    st.session_state.poster_tips = ""
    st.session_state.poster_quote = ""
    st.session_state.poster_org = "Your Organization Name"
    st.session_state.poster_helpline = "Your helpline / website here"


def cb_reset_all() -> None:
    first_cat = list(config.CATEGORIES)[0]
    preset = get_campaign_preset(first_cat)
    st.session_state.category = first_cat
    st.session_state.custom_category = ""
    st.session_state.user_prompt = config.CAMPAIGN_PRESETS[first_cat]["artwork_prompt"]
    st.session_state.poster_title = preset.headline
    st.session_state.poster_slogan = preset.slogan
    st.session_state.poster_body = preset.body
    st.session_state.poster_tips = "\n".join(preset.bullet_points)
    st.session_state.poster_quote = preset.quote
    st.session_state.poster_org = preset.org_name
    st.session_state.poster_helpline = preset.helpline
    st.session_state.poster_helpline = preset.helpline
    st.session_state.theme_choice = "✨ Auto-Detect from Prompt & Campaign"
    st.session_state.poster_layout = get_recommended_layout(first_cat)
    st.session_state.last_results = []
    st.session_state.last_request = None


def cb_regenerate() -> None:
    if st.session_state.last_request:
        req = dict(st.session_state.last_request)
        req["seed"] = None
        st.session_state.pending_request = req


def cb_clear_preview() -> None:
    st.session_state.last_results = []


def cb_delete_item(item_id: str) -> None:
    item = next((entry for entry in st.session_state.gallery if entry["id"] == item_id), None)
    if item and item.get("saved_path"):
        delete_saved_gallery_poster(item["saved_path"])
    st.session_state.gallery = [g for g in st.session_state.gallery if g["id"] != item_id]
    st.session_state.last_results = [g for g in st.session_state.last_results if g["id"] != item_id]


def cb_clear_gallery() -> None:
    try:
        for item in st.session_state.gallery:
            if item.get("saved_path"):
                delete_saved_gallery_poster(item["saved_path"])
    except (OSError, ValueError) as exc:
        st.error(f"Could not remove saved posters: {exc}")
        return
    st.session_state.gallery = []
    st.session_state.last_results = []


# ==================================================================
# Model helpers
# ==================================================================
@st.cache_resource(show_spinner=False)
def cached_device_info():
    if check_dependencies():
        return None
    return get_device_info()


@st.cache_resource(show_spinner=False, max_entries=1)
def load_generator(model_label: str, engine: str) -> WellnessImageGenerator:
    unload_cuda_memory()
    return WellnessImageGenerator(model_label, engine=engine).load()


def show_image(image, caption: str | None = None) -> None:
    st.image(image, caption=caption)


# ==================================================================
# Sidebar
# ==================================================================
def render_sidebar() -> None:
    with st.sidebar:
        c1, c2 = st.columns([1, 3])
        with c1:
            st.image(LOGO, width=52)
        with c2:
            st.markdown(f"<p class='sb-brand'>{config.APP_NAME}</p>"
                        f"<p class='sb-sub'>Campaign Poster Studio</p>", unsafe_allow_html=True)
        st.write("")

        st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
        st.divider()

        st.markdown("<div class='label'>Image generation</div>", unsafe_allow_html=True)
        if st.session_state.engine_mode == "Cloud inference":
            st.success("Hosted image model")
        else:
            st.success("Local diffusion")

        st.divider()

        # Device status
        info = cached_device_info()
        st.markdown("<div class='label'>Compute</div>", unsafe_allow_html=True)
        if st.session_state.engine_mode == "Cloud inference":
            st.info("Hosted image generation is active.")
        elif info is None:
            st.info("CPU available; local image generation may take longer.")
        elif info.device == "cuda":
            st.success(f"GPU ready: {info.name} ({info.vram_gb} GB)")
        elif info.device == "mps":
            st.success(f"GPU ready: {info.name}")
        else:
            st.info("Compute: CPU · local diffusion may be slow.")

        st.divider()
        st.metric("Posters created", len(st.session_state.gallery))
        st.markdown(f"<div class='disclaimer'>⚠️ {config.DISCLAIMER_SHORT}</div>", unsafe_allow_html=True)
        st.caption(f"{config.APP_TEAM} · Generative AI Healthcare Project")


# ==================================================================
# Home Page
# ==================================================================
def render_home() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <span class="badge">Patient-facing wellness awareness</span>
            <span class="badge">Editable copy, rendered separately</span>
            <h1>{config.APP_NAME}</h1>
            <h3>Wellness campaign visual generator</h3>
            <p>Create non-clinical campaign artwork, review the generated image, and add editable poster copy.
            All public-facing materials require internal medical, legal, and regulatory review.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    b1, b2, _ = st.columns([1.3, 1, 3])
    b1.button("Create a campaign", type="primary", **STRETCH, on_click=go_to, args=(PAGES[1],))
    b2.button("Open gallery", **STRETCH, on_click=go_to, args=(PAGES[2],))

    st.markdown("<div class='section-title'>A clear three-step workflow</div>", unsafe_allow_html=True)
    features = [
        ("01", "Campaign", "Choose a wellness theme and describe the artwork."),
        ("02", "Look", "Select a model, visual style, and colour palette."),
        ("03", "Poster text", "Edit the headline, supporting copy, organization, and contact details."),
    ]
    columns = st.columns(3)
    for column, (number, title, text) in zip(columns, features):
        column.markdown(
            f"<div class='card'><div class='icon'>{number}</div><h4>{title}</h4><p>{text}</p></div>",
            unsafe_allow_html=True,
        )


# ==================================================================
# Create Campaign Page
# ==================================================================
def _category_label(name: str) -> str:
    if name == config.CUSTOM_CATEGORY:
        return config.CUSTOM_CATEGORY
    return f"{config.CATEGORIES[name]['icon']} {name}"


def _style_label(name: str) -> str:
    return f"{config.STYLES[name]['icon']} {name}"


def _layout_label(name: str) -> str:
    item = config.POSTER_LAYOUTS[name]
    return f"{item['icon']} {item['name']}"


def render_create() -> None:
    st.markdown("## Campaign visual studio")
    st.markdown("<p class='muted'>Create non-clinical wellness artwork and add editable copy separately.</p>",
                unsafe_allow_html=True)

    left, right = st.columns([5, 6], gap="large")

    # ---------------- LEFT: inputs ----------------
    with left:
        with st.container(border=True):
            st.markdown("<div class='label'>Step 1 · Campaign</div>", unsafe_allow_html=True)
            camp_options = list(config.CATEGORIES) + [config.CUSTOM_CATEGORY]
            st.selectbox("Select Health Camp / Theme", camp_options, key="category",
                         on_change=cb_category_change,
                         format_func=_category_label, label_visibility="collapsed")

            st.text_input(
                "Optional custom theme",
                key="custom_category",
                placeholder="Add a short, non-clinical theme",
                help="Use general lifestyle and wellbeing language."
            )

            st.text_area(
                "Describe the artwork", key="user_prompt", height=95,
                max_chars=config.MAX_PROMPT_CHARS,
                placeholder="Describe the subject, activity, setting, and mood.",
                help="For portraits, describe one or two people in a waist-up view."
            )

            example_col, random_col = st.columns(2)
            example_col.button("Use example", on_click=cb_example_prompt, **STRETCH)
            random_col.button("Random suggestion", on_click=cb_random_prompt, **STRETCH)
            try:
                preview_bundle = build_prompt(
                    st.session_state.user_prompt,
                    st.session_state.category,
                    st.session_state.style,
                    st.session_state.custom_category,
                    audience=st.session_state.audience,
                )
                token_count = count_clip_tokens(preview_bundle.positive) + 2
                st.session_state.prompt_token_count = token_count
                st.progress(min(token_count, 77) / 77, text=f"Prompt tokens {token_count} / 77")
                if token_count > 77:
                    st.error(
                        "The artwork description alone exceeds CLIP's 77-token limit. "
                        "Shorten that description; poster copy doesn't use this token budget."
                    )
                else:
                    st.caption("This meter counts artwork prompt tokens only, not poster text.")
            except Exception as exc:
                st.session_state.prompt_token_count = None
                st.caption(f"CLIP tokenizer unavailable: {str(exc)[:120]}")

            st.markdown("<div class='label'>Step 2 · Look</div>", unsafe_allow_html=True)
            st.selectbox(
                "Audience",
                ["General adults", "Young adults", "Families", "Older adults", "Community members"],
                key="audience",
            )
            if st.session_state.engine_mode == "Cloud inference":
                st.caption("The hosted deployment selects its image model automatically.")
            else:
                st.selectbox("Local image model", list(config.MODELS), key="model_label")
                st.caption(config.MODELS[st.session_state.model_label]["description"])
            is_ai = True
            st.radio("Artwork Style", list(config.STYLES), key="style", format_func=_style_label,
                     horizontal=True, label_visibility="collapsed")

            st.selectbox("Poster layout", list(config.POSTER_LAYOUTS), key="poster_layout",
                         format_func=_layout_label, label_visibility="collapsed")
            st.caption(config.POSTER_LAYOUTS[st.session_state.poster_layout]["description"])
            color_mode = st.radio("Colour palette", ["Auto", "Custom", "Curated"],
                                  horizontal=True, label_visibility="collapsed")
            custom_pal = None
            if color_mode == "Custom":
                c_col1, c_col2 = st.columns(2)
                p_accent = c_col1.color_picker("Accent", value="#0D9488")
                p_dark = c_col2.color_picker("Dark tone", value="#0F172A")
                custom_pal = {
                    "primary": p_accent,
                    "accent": p_accent,
                    "accent2": "#F59E0B",
                    "bg_light": "#F8FAFC",
                    "bg_dark": p_dark,
                    "text_dark": p_dark,
                    "text_light": "#FFFFFF",
                    "badge_bg": p_accent,
                }
            elif color_mode == "Curated":
                st.selectbox("Curated palette", list(config.THEME_PALETTES),
                             key="theme_choice", label_visibility="collapsed")

            st.markdown("<div class='label'>Step 3 · Poster text</div>", unsafe_allow_html=True)
            st.caption("Poster text is rendered separately from the artwork prompt and wraps onto the poster.")
            st.text_input("Poster Headline", key="poster_title", max_chars=config.MAX_HEADLINE_CHARS,
                          placeholder="Enter a short headline")
            st.text_input("Slogan / Sub-headline", key="poster_slogan", max_chars=config.MAX_SLOGAN_CHARS,
                          placeholder="Optional supporting line")

            st.text_area("Campaign Educational Message", key="poster_body", height=120,
                         max_chars=config.MAX_BODY_CHARS,
                         placeholder="General lifestyle awareness copy; avoid clinical or outcome claims.")
            st.text_area("Health Tips (one per line, styled with campaign bullet markers)", key="poster_tips",
                         height=140, max_chars=config.MAX_TIPS_CHARS,
                         placeholder="Point 1\nPoint 2\nPoint 3\nPoint 4")

            c_q1, c_q2 = st.columns(2)
            with c_q1:
                st.text_input("Motivational quote", key="poster_quote", max_chars=config.MAX_QUOTE_CHARS,
                              placeholder="Optional")
            with c_q2:
                st.text_input("Organization name", key="poster_org", max_chars=config.MAX_ORG_NAME_CHARS)
            st.text_input("Helpline / website", key="poster_helpline", max_chars=config.MAX_HELPLINE_CHARS)

        with st.expander("⚙️ Advanced Generation Settings", expanded=False):
            res_label = st.selectbox(
                "Artwork Resolution", list(config.RESOLUTIONS),
                index=list(config.RESOLUTIONS).index(config.DEFAULT_RESOLUTION),
            )
            width, height = config.RESOLUTIONS[res_label]
            variations = st.slider("Variations", 1, config.MAX_VARIATIONS, 1)
            steps = config.MODELS[st.session_state.model_label]["default_steps"]
            if is_ai:
                model_cfg = config.MODELS[st.session_state.model_label]
                min_steps = 1 if model_cfg["max_steps"] <= 4 else 10
                steps = st.slider(
                    "Diffusion Steps",
                    min_steps,
                    model_cfg["max_steps"],
                    model_cfg["default_steps"],
                    key=f"steps_{st.session_state.model_label}",
                    help="More steps generally improve detail but take longer. SD-Turbo is intended for quick drafts.",
                )
                guidance = st.slider(
                    "Guidance scale", 0.0, 10.0,
                    float(model_cfg["default_guidance"]), step=0.5,
                )
            else:
                guidance = 0.0
            fixed_seed = st.toggle("Fixed Seed", value=False)
            seed = int(st.number_input("Seed Value", 0, 2**31 - 1, 42)) if fixed_seed else None
            save_disk = st.toggle("Save PNGs to generated_images folder", value=True)

        g1, g2 = st.columns([3, 1])
        generate_clicked = g1.button("🚀 Generate Campaign Poster", type="primary", **STRETCH)
        g2.button("↺ Reset", on_click=cb_reset_all, **STRETCH)

        if generate_clicked:
            st.session_state.first_visit = False
            effective_category = st.session_state.custom_category.strip() if st.session_state.custom_category.strip() else st.session_state.category
            result = validate_user_input(
                st.session_state.user_prompt,
                effective_category,
                poster_text=" ".join(
                    [
                        st.session_state.poster_title, st.session_state.poster_slogan,
                        st.session_state.poster_body, st.session_state.poster_tips,
                        st.session_state.poster_quote, st.session_state.poster_org,
                        st.session_state.poster_helpline,
                    ]
                ),
            )
            token_count = st.session_state.get("prompt_token_count")
            if token_count is not None and token_count > 77:
                result.errors.append(
                    "The artwork description alone exceeds CLIP's 77-token limit. "
                    "Shorten that description; poster copy doesn't use this token budget."
                )
                result.ok = False
            for err in result.errors:
                st.error(err)
            for warn in result.warnings:
                st.warning(warn)

            if result.ok:
                st.session_state.pending_request = {
                    "user_prompt": st.session_state.user_prompt,
                    "category": effective_category,
                    "custom_category": st.session_state.custom_category,
                    "poster_layout": st.session_state.poster_layout,
                    "theme_choice": st.session_state.get("theme_choice", "✨ Auto-Detect from Prompt & Campaign"),
                    "custom_palette": custom_pal,
                    "poster_title": st.session_state.poster_title,
                    "poster_slogan": st.session_state.poster_slogan,
                    "poster_body": st.session_state.poster_body,
                    "poster_tips": st.session_state.poster_tips,
                    "poster_quote": st.session_state.poster_quote,
                    "poster_org": st.session_state.poster_org,
                    "poster_helpline": st.session_state.poster_helpline,
                    "style": st.session_state.style,
                    "audience": st.session_state.audience,
                    "engine_mode": st.session_state.engine_mode,
                    "model_label": st.session_state.model_label,
                    "width": width, "height": height,
                    "steps": steps,
                    "guidance": guidance,
                    "variations": variations, "seed": seed,
                    "add_footer": True, "save_disk": save_disk,
                }

    # ---------------- RIGHT: generation + preview ----------------
    with right:
        if st.session_state.pending_request:
            request = st.session_state.pending_request
            st.session_state.pending_request = None
            run_generation(request)
        render_preview()


def run_generation(request: dict, add_to_gallery: bool = True, notify: bool = True) -> bool:
    try:
        bundle = build_prompt(
            request["user_prompt"], request["category"], request["style"],
            request.get("custom_category", ""),
            audience=request.get("audience", ""),
        )
    except Exception as exc:
        st.error(f"Could not prepare the CLIP prompt. Check tokenizer access and try again: {str(exc)[:180]}")
        return False
    prompt_token_count = count_clip_tokens(bundle.positive) + 2
    if prompt_token_count > 77:
        st.error(
            "The artwork description alone exceeds CLIP's 77-token limit. "
            "Shorten that description; poster copy doesn't use this token budget."
        )
        return False

    # Prepare structured poster copy
    bullet_list = [line.strip("-•* ").strip() for line in request.get("poster_tips", "").splitlines() if line.strip()]
    if not bullet_list:
        preset = get_campaign_preset(bundle.category)
        bullet_list = preset.bullet_points

    copy = PosterCopy(
        headline=request.get("poster_title", "").strip() or bundle.category.upper(),
        slogan=request.get("poster_slogan", "").strip(),
        body=request.get("poster_body", "").strip(),
        bullet_points=bullet_list,
        quote=request.get("poster_quote", "").strip(),
        org_name=request.get("poster_org", "").strip() or "Your Organization Name",
        helpline=request.get("poster_helpline", "").strip() or "Your helpline / website here",
    )
    claim_flags = lint_poster_copy(copy)
    if claim_flags:
        st.warning(
            "Poster copy contains potentially risky health claims "
            f"({', '.join(claim_flags)}). Review and edit the copy before export."
        )

    use_instant = request.get("engine_mode") == "Instant vector draft"
    n = request["variations"]

    with st.container(border=True):
        st.markdown("#### ⏳ Creating your Campaign Poster…")
        generator = None
        if not use_instant:
            try:
                status = (
                    "Waking up the hosted image model..."
                    if request["engine_mode"] == "Cloud inference"
                    else "Loading local diffusion model..."
                )
                with st.spinner(status):
                    generator = load_generator(request["model_label"], request["engine_mode"])
            except GenerationError as exc:
                st.error(f"Generative AI loading error: {exc}")
                return False

        candidate_count = 4 if re.search(
            r"\b(woman|women|man|person|people|human|face|portrait|doctor|nurse|dentist|"
            r"optometrist|patient|child|senior|elder|family|adult|professional|girl|boy)\b",
            bundle.positive,
            re.IGNORECASE,
        ) else 3
        total_candidates = candidate_count * n
        bar = st.progress(0.0, text=f"Generating candidates 1/{total_candidates}")

        def on_progress(index: int, total_candidates: int, stage: str, step: int, total_steps: int) -> None:
            done = (index + step / max(1, total_steps)) / total_candidates
            text = f"{stage} {index + 1}/{total_candidates}"
            if stage == "Generating":
                text += f" · step {step}/{total_steps}"
            bar.progress(min(done, 1.0), text=text)

        try:
            spinner_message = (
                "Waking up the hosted image model..."
                if request["engine_mode"] == "Cloud inference"
                else "Generating candidates and checking for stray text / clear faces..."
            )
            with st.spinner(spinner_message):
                if use_instant:
                    # Instant Graphic Engine
                    generator = WellnessImageGenerator(
                        request["model_label"], engine=request["engine_mode"]
                    )
                    outputs = generator.generate(
                        prompt=bundle.positive, negative_prompt=bundle.negative,
                        width=request["width"], height=request["height"],
                        num_images=n, seed=request["seed"], category=bundle.category,
                        use_instant_engine=True,
                        theme_choice=request.get("theme_choice"),
                    )
                else:
                    outputs = generator.generate(
                        prompt=bundle.positive, negative_prompt=bundle.negative,
                        width=request["width"], height=request["height"],
                        num_images=n, seed=request["seed"], category=bundle.category,
                        steps=request.get("steps", 2),
                        guidance=request.get("guidance"),
                        progress_callback=on_progress,
                        theme_choice=request.get("theme_choice"),
                    )
        except Exception as exc:
            bar.empty()
            st.error(f"Generation error: {exc}")
            return False

        bar.empty()

    # --- Composite into reference poster layout ---
    items = []
    for out in outputs:
        poster_img = image_utils.create_campaign_poster(
            out.image, copy, category=bundle.category,
            layout_name=request["poster_layout"],
            user_prompt=request.get("user_prompt", ""),
            theme_choice=request.get("theme_choice"),
            custom_palette=request.get("custom_palette"),
        )
        footer_text = (
            "Procedurally generated, non-clinical wellness illustration. Requires internal "
            "medical/legal/regulatory (MLR) review before public use."
            if out.engine_label == "Built-in vector artwork"
            else config.DISCLAIMER_SHORT
        )
        poster_img = image_utils.add_disclaimer_footer(poster_img, text=footer_text)

        preset_campaign = request.get("preset_campaign", False)
        filename = (
            f"{image_utils.slugify(bundle.category)}.png"
            if preset_campaign else image_utils.make_filename(bundle.category, out.seed)
        )
        saved = image_utils.save_image(
            poster_img,
            filename,
            output_dir=config.PRESET_POSTER_DIR if preset_campaign else None,
        ) if request["save_disk"] else None

        items.append({
            "id": f"preset-{image_utils.slugify(bundle.category)}" if preset_campaign else uuid.uuid4().hex,
            "png_bytes": image_utils.pil_to_png_bytes(poster_img),
            "artwork_png_bytes": image_utils.pil_to_png_bytes(out.image.convert("RGB")),
            "filename": filename,
            "saved_path": str(saved) if saved else None,
            "category": bundle.category,
            "layout": request["poster_layout"],
            "style": bundle.style,
            "user_prompt": bundle.user_text,
            "poster_title": copy.headline,
            "poster_slogan": copy.slogan,
            "poster_body": copy.body,
            "final_prompt": bundle.positive,
        if any(out.engine_label == "Built-in vector artwork" for out in outputs):
            st.warning(
                "Hugging Face returned HTTP 402 because hosted image inference requires paid credits. "
                "One or more requested posters use WellAware AI's built-in vector illustration instead; "
                "they are not photos or images generated by a hosted AI model."
            )
            "negative_prompt": bundle.negative,
            "token_count": prompt_token_count,
            "readable": bundle.readable,
            "seed": out.seed,
            "text_gate_result": out.text_gate_result,
            "face_gate_result": out.face_gate_result,
            "seconds": out.seconds,
            "size": f"{poster_img.width}×{poster_img.height}",
            "engine": (
                "🎨 Built-in Vector Artwork (free fallback)"
                if out.engine_label == "Built-in vector artwork"
                else "⚡ Fast Draft Engine" if use_instant
                else f"🤖 Generative AI ({request['model_label'].split()[0]})"
            ),
            "created": datetime.now().strftime("%d %b %Y, %H:%M:%S"),
            "preset_campaign": preset_campaign,
        })
        image_utils.append_audit_record({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "category": bundle.category,
            "user_prompt": bundle.user_text,
            "final_prompt": bundle.positive,
            "negative_prompt": bundle.negative,
            "seed": out.seed,
            "model": out.engine_label or request["model_label"],
            "text_gate_result": out.text_gate_result,
            "face_gate_result": out.face_gate_result,
        })
        if out.text_gate_result.startswith("stray text"):
            st.warning("The artwork still contains possible stray text after retries; review it before use.")
        if out.face_gate_result.startswith("failed"):
            st.warning("Face quality check did not pass; review or regenerate this artwork.")
        if out.face_gate_result.startswith("unavailable"):
            st.warning("Face detector is unavailable; face quality was not verified.")

    if request.get("preset_campaign") and not items:
        st.error(f"No poster was returned for {bundle.category}.")
        return False
    if add_to_gallery:
        st.session_state.last_results = items
        st.session_state.last_request = request
        st.session_state.gallery = items + st.session_state.gallery
    if notify:
        st.toast(f"✅ {len(items)} campaign poster(s) created!", icon="🎉")
    return True


def render_preview() -> None:
    results = st.session_state.last_results
    if not results:
        st.markdown(
            "<div class='empty'><div class='big'>🖼️</div>"
            "<h4>Your finished poster will appear here</h4>"
            "<p>Select your theme & layout on the left, then click <b>Generate Campaign Poster</b>.</p></div>",
            unsafe_allow_html=True,
        )
        return

    first = results[0]
    with st.container(border=True):
        st.markdown("#### Preview")
        st.markdown(f"<span class='chip'>🏷️ {first['category']}</span>"
                    f"<span class='chip'>📐 {first.get('layout', 'Editorial')}</span>"
                    f"<span class='chip'>⚡ {first.get('engine', 'Engine')}</span>"
                    f"<span class='chip'>📏 {first['size']} px</span>", unsafe_allow_html=True)

        poster_tab, art_tab, details_tab = st.tabs(["Poster", "Artwork only", "How this was made"])
        with poster_tab:
            if len(results) == 1:
                show_image(first["png_bytes"], caption=f"{first['poster_title']} · {first['seconds']}s")
            else:
                grid = st.columns(len(results))
                for idx, item in enumerate(results):
                    with grid[idx]:
                        show_image(item["png_bytes"], caption=f"Variation {idx + 1}")
            dl_png, dl_jpg = st.columns(2)
            dl_png.download_button(
                "Download PNG", data=first["png_bytes"], file_name=first["filename"],
                mime="image/png", key=f"dl_prev_{first['id']}", type="primary", **STRETCH,
            )
            poster_pil = image_utils.png_bytes_to_pil(first["png_bytes"])
            dl_jpg.download_button(
                "Download JPG", data=image_utils.pil_to_jpeg_bytes(poster_pil),
                file_name=first["filename"].rsplit(".", 1)[0] + ".jpg",
                mime="image/jpeg", key=f"dl_jpg_{first['id']}", **STRETCH,
            )
        with art_tab:
            show_image(first["artwork_png_bytes"])
            st.download_button(
                "Download artwork PNG", data=first["artwork_png_bytes"],
                file_name="wellaware_artwork.png", mime="image/png",
                key=f"dl_art_{first['id']}", **STRETCH,
            )
        with details_tab:
            st.markdown(f"**Summary:** {first['readable']}")
            st.markdown(f"**CLIP tokens:** {first['token_count']} / 77")
            st.markdown(f"**Seed:** {first['seed']}")
            st.markdown(f"**Text gate:** {first['text_gate_result']}")
            st.markdown(f"**Face gate:** {first['face_gate_result']}")
            st.markdown("**Positive prompt**")
            st.code(first["final_prompt"], language=None)
            st.markdown("**Negative prompt**")
            st.code(first["negative_prompt"], language=None)

        r1, r2 = st.columns(2)
        r1.button("🔄 Regenerate", on_click=cb_regenerate, **STRETCH)
        r2.button("🧹 Clear Preview", on_click=cb_clear_preview, **STRETCH)

    st.markdown(f"<div class='disclaimer'>⚠️ {config.DISCLAIMER_FULL.replace('**', '')}</div>",
                unsafe_allow_html=True)


# ==================================================================
# Gallery Page
# ==================================================================
def _preset_generation_request(category: str) -> dict:
    preset = config.CAMPAIGN_PRESETS[category]
    model = config.MODELS[config.DEFAULT_MODEL]
    width, height = config.RESOLUTIONS[config.DEFAULT_RESOLUTION]
    seed = int.from_bytes(hashlib.sha256(category.encode("utf-8")).digest()[:4], "big") & 0x7FFFFFFF
    return {
        "user_prompt": preset["artwork_prompt"],
        "category": category,
        "custom_category": "",
        "poster_layout": config.DEFAULT_POSTER_LAYOUT,
        "theme_choice": "✨ Auto-Detect from Prompt & Campaign",
        "custom_palette": None,
        "poster_title": preset["headline"],
        "poster_slogan": preset["slogan"],
        "poster_body": preset["body"],
        "poster_tips": "\n".join(preset["bullet_points"]),
        "poster_quote": preset["quote"],
        "poster_org": preset["org_name"],
        "poster_helpline": preset["helpline"],
        "style": config.DEFAULT_STYLE,
        "audience": "General adults",
        "engine_mode": st.session_state.engine_mode,
        "model_label": config.DEFAULT_MODEL,
        "width": width,
        "height": height,
        "steps": model["default_steps"],
        "guidance": model["default_guidance"],
        "variations": 1,
        "seed": seed,
        "add_footer": True,
        "save_disk": True,
        "preset_campaign": True,
    }


def missing_preset_categories() -> list[str]:
    created_categories = {
        item["category"]
        for item in st.session_state.gallery
        if item.get("preset_campaign")
    }
    return [category for category in config.CAMPAIGN_PRESETS if category not in created_categories]


def generate_missing_preset_campaigns() -> bool:
    missing = missing_preset_categories()
    if not missing:
        return True

    progress = st.progress(0.0, text=f"Starting campaign batch · {len(st.session_state.gallery)} poster(s) in gallery")
    completed_all = True
    for index, category in enumerate(missing):
        progress.progress(
            index / len(missing),
            text=f"Generating preset {index + 1}/{len(missing)} · {category}",
        )
        request = _preset_generation_request(category)
        with st.status(f"Generating {category}", expanded=False) as status:
            try:
                succeeded = run_generation(request, notify=False)
            except Exception as exc:
                st.error(f"Could not finish {category}: {exc}")
                succeeded = False
            status.update(
                label=f"{category} {'completed' if succeeded else 'failed'}",
                state="complete" if succeeded else "error",
            )
        if not succeeded:
            completed_all = False
            st.warning(
                "The batch stopped at the first error. Finished posters have been added to the gallery; "
                "run the batch again later to continue."
            )
            break

    progress.empty()
    return completed_all


def render_gallery() -> None:
    gallery = st.session_state.gallery
    st.markdown("## Poster Gallery")
    st.markdown("<p class='muted'>Your campaign posters and generated preset campaigns appear together here.</p>",
                unsafe_allow_html=True)

    if missing_preset_categories():
        if st.button("Generate missing preset campaigns", type="primary", **STRETCH):
            if generate_missing_preset_campaigns():
                st.rerun()
        st.caption(
            "Preset campaigns are generated with SDXL and saved as they finish. "
            "If a batch stops, use this button again to continue."
        )

    gallery = st.session_state.gallery
    if not gallery:
        st.markdown("<div class='empty'><div class='big'>📭</div><h4>No saved posters yet</h4>"
                    "<p>Create a poster or generate the preset campaigns above to fill this gallery.</p></div>",
                    unsafe_allow_html=True)
        st.write("")
        st.button("✨ Create your first poster", type="primary", on_click=go_to, args=(PAGES[1],))
        return

    t1, t2, t3, t4 = st.columns([3, 1.3, 1.5, 1])
    categories = sorted({g["category"] for g in gallery})
    selected = t1.multiselect("Filter by category", categories, placeholder="All categories")
    t2.download_button("📦 Download All (ZIP)", data=image_utils.make_zip(gallery),
                       file_name="wellaware_posters.zip", mime="application/zip",
                       **STRETCH)
    t3.download_button(
        "Audit log (CSV)", data=image_utils.audit_log_bytes(),
        file_name="auditlog.csv", mime="text/csv", **STRETCH,
    )
    t4.button("Clear gallery", on_click=cb_clear_gallery, **STRETCH)

    items = [g for g in gallery if not selected or g["category"] in selected]
    st.caption(f"Showing {len(items)} of {len(gallery)} poster(s)")

    cols_per_row = 3
    for row_start in range(0, len(items), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, item in zip(cols, items[row_start:row_start + cols_per_row]):
            with col, st.container(border=True):
                show_image(item["png_bytes"])
                st.markdown(f"<span class='chip'>🏷️ {item['category']}</span>"
                            f"<span class='chip'>📐 {item.get('layout', 'Editorial')}</span>", unsafe_allow_html=True)
                st.markdown(f"**{item.get('poster_title', 'Poster')}**")
                d1, d2 = st.columns([3, 1])
                d1.download_button("⬇️ PNG", data=item["png_bytes"], file_name=item["filename"],
                                   mime="image/png", key=f"dl_gal_{item['id']}", **STRETCH)
                d2.button("🗑️", key=f"del_{item['id']}", on_click=cb_delete_item, args=(item["id"],),
                          help="Remove", **STRETCH)


# ==================================================================
# Guidelines & Help Page
# ==================================================================
def render_help() -> None:
    st.markdown("## Guidelines")
    st.markdown(f"<div class='disclaimer'>⚠️ {config.DISCLAIMER_FULL.replace('**', '')}</div>",
                unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("#### Workflow and capabilities")
        st.markdown("- Two flexible poster layouts with artwork generated separately from text\n"
                    "- Editable headline, slogan, body, organization, and contact fields\n"
                    "- OCR or edge-density text checks and an optional face-quality signal")
    with c2, st.container(border=True):
        st.markdown("#### Content restrictions")
        st.markdown("- No diagnostic scans (X-rays, MRIs, biopsies)\n"
                    "- No pharma brand logos or commercial endorsements\n"
                    "- Clinical and dosage-related prompts are blocked; risky copy is flagged\n"
                    "- Non-clinical awareness visuals only")

    st.markdown("<div class='section-title'>Before you publish</div>", unsafe_allow_html=True)
    st.markdown(
        "- Review all generated artwork and editable copy before export.\n"
        "- Add only organization names and contact details that you are authorized to publish.\n"
        "- Obtain internal medical, legal, and regulatory review before public use."
    )


# ==================================================================
# Main
# ==================================================================
def main() -> None:
    inject_css()
    init_state()
    render_sidebar()

    page = st.session_state.nav
    if page == PAGES[0]:
        render_home()
    elif page == PAGES[1]:
        render_create()
    elif page == PAGES[2]:
        render_gallery()
    else:
        render_help()

    st.markdown(f"<div class='footer'>{config.APP_NAME} · {config.APP_SUBTITLE} · "
                f"{config.APP_TEAM} · Non-clinical illustrative posters.</div>",
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
