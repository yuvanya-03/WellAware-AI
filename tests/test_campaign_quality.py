import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image, ImageFont

import config
from modules import image_utils
from modules.image_utils import _readable_text_color, create_campaign_poster
from modules.image_generator import ProviderPaymentError, WellnessImageGenerator
from modules.preset_campaigns import load_saved_gallery_posters, load_saved_preset_posters
from modules.prompt_processor import (
    PosterCopy,
    build_prompt,
    count_clip_tokens,
    lint_poster_copy,
    sanitize_positive_prompt,
    truncate_prompt_by_priority,
    validate_user_input,
)


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return re.findall(r"[A-Za-z0-9]+", text)


class CampaignQualityTests(unittest.TestCase):
    def test_paid_provider_402_falls_back_to_free_vector_art(self):
        generator = WellnessImageGenerator(engine="Cloud inference")
        generated_art = Image.new("RGB", (192, 256), "#0D9488")

        with (
            patch.object(generator, "load", return_value=generator),
            patch.object(
                generator, "_generate_cloud",
                side_effect=ProviderPaymentError("HTTP 402"),
            ) as hosted_generation,
            patch.object(
                image_utils, "generate_procedural_vector_art",
                return_value=generated_art,
            ) as vector_generation,
        ):
            outputs = generator.generate(
                prompt="One adult walking in a sunny park",
                negative_prompt="text",
                width=192,
                height=256,
                num_images=2,
                seed=41,
            )

        self.assertEqual(len(outputs), 2)
        self.assertEqual(hosted_generation.call_count, 1)
        self.assertEqual(vector_generation.call_count, 2)
        self.assertTrue(all(out.engine_label == "Built-in vector artwork" for out in outputs))
        self.assertTrue(all("not applicable" in out.text_gate_result for out in outputs))

    def test_cloud_provider_402_stops_paid_retries(self):
        generator = WellnessImageGenerator(engine="Cloud inference")
        client = Mock()
        payment_error = RuntimeError("402 Payment Required")
        payment_error.response = SimpleNamespace(status_code=402)
        client.text_to_image.side_effect = payment_error
        hub = SimpleNamespace(InferenceClient=Mock(return_value=client))

        with (
            patch.dict(os.environ, {"HF_TOKEN": "test-token"}),
            patch.dict(sys.modules, {"huggingface_hub": hub}),
        ):
            with self.assertRaises(ProviderPaymentError):
                generator._generate_cloud("prompt", "negative", 512, 512, 4, 1.0, 7)

        self.assertEqual(client.text_to_image.call_count, 1)

    def test_saved_preset_posters_load_for_unified_gallery(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            poster_dir = Path(temporary_directory)
            image_utils.save_image(
                Image.new("RGB", (320, 480), "#0D9488"),
                "mental-health.png",
                output_dir=poster_dir,
            )

            posters = load_saved_preset_posters(poster_dir)

        self.assertEqual(list(posters), ["Mental Health"])
        self.assertEqual(posters["Mental Health"]["filename"], "mental-health.png")
        self.assertEqual(posters["Mental Health"]["size"], "320×480")
        self.assertEqual(len(config.CAMPAIGN_PRESETS), 15)

    def test_existing_generated_pngs_load_into_the_gallery(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            image_utils.save_image(
                Image.new("RGB", (320, 480), "#0D9488"),
                "wellaware_mental-health_20261125_000001_123.png",
                output_dir=output_dir,
            )

            posters = load_saved_gallery_posters(output_dir)

        self.assertEqual(len(posters), 1)
        self.assertEqual(posters[0]["category"], "Mental Health")
        self.assertEqual(posters[0]["poster_title"], config.CAMPAIGN_PRESETS["Mental Health"]["headline"])

    def test_sanitizer_removes_negation_and_lettering_terms_from_all_prompt_sources(self):
        sources = [config.QUALITY_TAGS]
        sources.extend(preset["artwork_prompt"] for preset in config.CAMPAIGN_PRESETS.values())
        sources.extend(category["visuals"] for category in config.CATEGORIES.values())
        sources.extend(style["prompt"] for style in config.STYLES.values())
        for source in sources:
            sanitized = sanitize_positive_prompt(source).lower()
            for term in (
                "no", "without", "not", "avoid", "poster", "dribbble", "banner",
                "label", "sign", "chart", "infographic", "typography", "logo",
                "vision chart", "healthcare graphic", "award-winning design",
            ):
                self.assertIsNone(
                    re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", sanitized),
                    f"{term!r} survived in {source!r}",
                )

    def test_prompt_sanitization_and_negative_prompt(self):
        bundle = build_prompt(
            "A poster with no text showing one smiling woman, without signs",
            "General Wellness",
            "Photographic & Realistic",
            tokenizer=WordTokenizer(),
        )
        self.assertNotRegex(bundle.positive.lower(), r"\b(no|without|poster|text|signs?)\b")
        for token in (
            "text", "letters", "words", "numbers", "caption", "signature",
            "watermark", "logo", "sign", "label", "banner", "gibberish",
            "handwriting", "qr code", "ui", "border", "frame",
        ):
            self.assertIn(token, bundle.negative)
        self.assertLessEqual(count_clip_tokens(bundle.negative, WordTokenizer()), 75)
        self.assertIn("face clearly visible", bundle.positive)

        model_bundle = build_prompt(
            "One smiling woman in a bright garden",
            "General Wellness",
            "Photographic & Realistic",
        )
        self.assertLessEqual(count_clip_tokens(model_bundle.negative), 75)

    def test_before_after_is_rejected_only_as_artwork_imagery(self):
        image_request = validate_user_input(
            "A before and after comparison of an active adult in a park"
        )
        self.assertFalse(image_request.ok)
        self.assertIn("Before/after comparison imagery isn't supported", image_request.errors[0])

        copy_text = validate_user_input(
            "A relaxed adult walking in a park",
            poster_text="A note about moments before and after a walk.",
        )
        self.assertTrue(copy_text.ok)

    def test_safety_filter_uses_word_boundaries_and_blocks_dosage_claims(self):
        allowed = validate_user_input("A smiling family walking in a green park")
        self.assertTrue(allowed.ok)

        blocked = validate_user_input("Lipitor 20mg pills curing cholesterol")
        self.assertFalse(blocked.ok)
        self.assertTrue(any("Lipitor" in error or "dosage" in error for error in blocked.errors))

    def test_token_truncation_preserves_subject_and_drops_lower_priority(self):
        subject = "one adult walking in a sunny park"
        result = truncate_prompt_by_priority(
            subject,
            "photorealistic editorial portrait",
            "teal aqua and golden colors",
            "very detailed extras " * 40,
            tokenizer=WordTokenizer(),
            max_tokens=20,
        )
        self.assertTrue(result.startswith(subject))
        self.assertNotIn("very detailed extras", result)
        self.assertLessEqual(len(WordTokenizer().encode(result)), 20)

    def test_audience_is_retained_within_clip_context_limit(self):
        bundle = build_prompt(
            "One friendly healthcare professional in a bright modern clinic.",
            "General Wellness",
            "Photographic & Realistic",
            audience="Families",
        )
        self.assertIn("families", bundle.positive.lower())
        self.assertLessEqual(count_clip_tokens(bundle.positive) + 2, 77)

    def test_people_guidance_does_not_push_short_subject_over_clip_limit(self):
        bundle = build_prompt(
            "One friendly healthcare professional in a bright modern clinic.",
            "General Wellness",
            "Photographic & Realistic",
        )
        self.assertIn("face clearly visible", bundle.positive)
        self.assertLessEqual(count_clip_tokens(bundle.positive) + 2, 77)

    def test_claim_linter_flags_copy_and_presets_are_neutral(self):
        copy = PosterCopy(
            headline="WELLNESS",
            slogan="",
            body="This daily routine saves lives.",
            bullet_points=[],
            quote="",
            org_name="Your Organization Name",
            helpline="Your helpline / website here",
        )
        self.assertIn("saves lives", lint_poster_copy(copy))
        for preset in config.CAMPAIGN_PRESETS.values():
            self.assertEqual(preset["org_name"], "Your Organization Name")
            self.assertEqual(preset["helpline"], "Your helpline / website here")
            self.assertEqual(lint_poster_copy(PosterCopy(
                preset["headline"], preset["slogan"], preset["body"],
                preset["bullet_points"], preset["quote"],
                preset["org_name"], preset["helpline"],
            )), [])

    def test_layout_wraps_extreme_copy_without_fixed_canvas_overflow(self):
        long_text = "A thoughtful everyday wellbeing choice " * 80
        long_copy = PosterCopy(
            headline="A LONG WELLNESS HEADLINE " * 5,
            slogan="A supportive line " * 8,
            body=long_text,
            bullet_points=[long_text, "Short tip"],
            quote=long_text,
            org_name="Your Organization Name",
            helpline="Your helpline / website here",
        )
        short_copy = PosterCopy(
            headline="WELLNESS",
            slogan="Everyday care",
            body="Small moments of rest and connection.",
            bullet_points=["Choose movement you enjoy"],
            quote="Take time for yourself.",
            org_name="Your Organization Name",
            helpline="Your helpline / website here",
        )
        artwork = Image.new("RGB", (192, 256), "#5AAABB")
        with patch.object(
            image_utils, "_load_font",
            side_effect=lambda size, **kwargs: ImageFont.load_default(size=max(8, size)),
        ):
            for layout_name in config.POSTER_LAYOUTS:
                poster = create_campaign_poster(
                    artwork, long_copy, category="General Wellness", layout_name=layout_name,
                )
                self.assertEqual(poster.width, config.POSTER_WIDTH)
                self.assertGreater(poster.height, config.POSTER_HEIGHT)

                short_poster = create_campaign_poster(
                    artwork, short_copy, category="General Wellness", layout_name=layout_name,
                )
                self.assertEqual(short_poster.width, config.POSTER_WIDTH)
                self.assertEqual(short_poster.height, config.POSTER_HEIGHT)
        self.assertEqual(_readable_text_color("#BFE8F3", "#FFFFFF"), "#0B1D30")


if __name__ == "__main__":
    unittest.main()
