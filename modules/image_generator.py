"""Local Diffusers and Hugging Face Inference API image generation."""

from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import config


class GenerationError(Exception):
    """Raised with a user-friendly message when generation fails."""


@dataclass
class DeviceInfo:
    device: str          # "cuda", "mps" or "cpu"
    name: str            # e.g. "Tesla T4"
    dtype_name: str      # "float16" or "float32"
    vram_gb: Optional[float] = None

    @property
    def is_gpu(self) -> bool:
        return self.device in ("cuda", "mps")


@dataclass
class GeneratedImage:
    image: "object"      # PIL.Image.Image
    seed: int
    seconds: float
    text_gate_result: str = "not checked"
    face_gate_result: str = "not checked"


# ------------------------------------------------------------------
# Environment checks
# ------------------------------------------------------------------
def check_dependencies() -> Optional[str]:
    missing = []
    for pkg in ("torch", "diffusers", "transformers", "accelerate"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        return ("Missing Python packages: " + ", ".join(missing) +
                ". Install them with `pip install -r requirements.txt`.")
    return None


def get_device_info() -> DeviceInfo:
    import torch

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return DeviceInfo("cuda", props.name, "float16", round(props.total_memory / 1024 ** 3, 1))
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return DeviceInfo("mps", "Apple Silicon GPU (MPS)", "float32")
    return DeviceInfo("cpu", "CPU", "float32")


def _get_hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None


# ------------------------------------------------------------------
# Generator class
# ------------------------------------------------------------------
class WellnessImageGenerator:
    """Wraps Diffusers pipeline + Fast Vector Graphic Engine."""

    def __init__(self, model_label: str = config.DEFAULT_MODEL, engine: str = "Local GPU"):
        if model_label not in config.MODELS:
            model_label = config.DEFAULT_MODEL
        self.model_label = model_label
        self.model_cfg = config.MODELS[model_label]
        self.engine = engine
        self.device_info: Optional[DeviceInfo] = None
        self.pipe = None

    # ---------------- loading ----------------
    def load(self) -> "WellnessImageGenerator":
        if self.engine == "Cloud inference":
            if not _get_hf_token():
                raise GenerationError(
                    "Cloud inference needs an HF_TOKEN secret. Add it to the environment or Space settings."
                )
            return self
        dep_error = check_dependencies()
        if dep_error:
            raise GenerationError(dep_error)

        import torch
        from diffusers import AutoPipelineForText2Image

        self.device_info = get_device_info()
        dtype = torch.float16 if self.device_info.dtype_name == "float16" else torch.float32
        repo_id = self.model_cfg["repo_id"]

        load_kwargs = {"torch_dtype": dtype, "use_safetensors": True}
        token = _get_hf_token()
        if token:
            load_kwargs["token"] = token
        if dtype == torch.float16:
            load_kwargs["variant"] = "fp16"
        vae_repo = self.model_cfg.get("vae_repo_id")
        if vae_repo:
            try:
                from diffusers import AutoencoderKL
                load_kwargs["vae"] = AutoencoderKL.from_pretrained(
                    vae_repo, torch_dtype=dtype, use_safetensors=True
                )
            except Exception as exc:
                raise GenerationError(
                    f"Could not load the SDXL VAE '{vae_repo}': {str(exc)[:220]}"
                ) from exc

        try:
            try:
                pipe = AutoPipelineForText2Image.from_pretrained(repo_id, **load_kwargs)
            except (OSError, ValueError):
                load_kwargs.pop("variant", None)
                pipe = AutoPipelineForText2Image.from_pretrained(repo_id, **load_kwargs)
        except Exception as exc:
            raise GenerationError(
                f"Could not load AI model '{repo_id}'. Check internet connection and disk space. "
                f"Details: {str(exc)[:250]}"
            ) from exc

        try:
            if self.device_info.device == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to(self.device_info.device)
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
        except Exception as exc:
            raise GenerationError(f"Could not prepare model on {self.device_info.device}: {exc}") from exc

        try:
            from diffusers import DPMSolverMultistepScheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True,
            )
        except (ImportError, ValueError, TypeError):
            pass

        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        return self

    # ---------------- generation ----------------
    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        width: int = 512,
        height: int = 512,
        num_images: int = 1,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        category: str = "General Wellness",
        progress_callback: Optional[Callable[[int, int, str, int, int], None]] = None,
        use_instant_engine: bool = False,
        theme_choice: Optional[str] = None,
    ) -> List[GeneratedImage]:
        """
        Generate `num_images` variations.
        If use_instant_engine is True, generates rich, themed vector art in 0.1s.
        """
        from modules.prompt_processor import sanitize_positive_prompt
        prompt = sanitize_positive_prompt(prompt)
        if not prompt:
            raise GenerationError("Prompt has no visual description after removing lettering instructions.")
        base_seed = seed if seed is not None else random.randint(0, 2 ** 31 - 1)
        num_images = max(1, min(int(num_images), config.MAX_VARIATIONS))

        if use_instant_engine:
            # Fast Layout Preview Synthesizer
            from modules.image_utils import generate_procedural_vector_art
            results = []
            for i in range(num_images):
                img_seed = base_seed + i
                start = time.time()
                img = generate_procedural_vector_art(category, prompt, width, height, theme_choice=theme_choice)
                results.append(GeneratedImage(image=img.convert("RGB"), seed=img_seed, seconds=round(time.time() - start, 2)))
            return results

        if self.engine == "Cloud inference":
            self.load()
        # Ensure local AI pipeline is loaded
        if self.pipe is None:
            if self.engine != "Cloud inference":
                self.load()

        steps = int(steps or self.model_cfg["default_steps"])
        guidance = float(self.model_cfg["default_guidance"] if guidance is None else guidance)
        width, height = (int(width) // 8) * 8, (int(height) // 8) * 8

        from modules.image_utils import assess_face_quality, detect_stray_text

        people_requested = any(
            re.search(r"\b" + re.escape(keyword) + r"\b", prompt, re.IGNORECASE)
            for keyword in (
                "woman", "women", "man", "person", "people", "human", "face",
                "portrait", "doctor", "nurse", "dentist", "optometrist", "patient",
                "child", "senior", "elder", "family", "adult", "professional",
                "girl", "boy", "mother", "father",
            )
        )
        results: List[GeneratedImage] = []
        candidates_per_result = 4 if people_requested else 3
        candidate_total = num_images * candidates_per_result
        for i in range(num_images):
            candidate_count = candidates_per_result
            candidates = []
            for attempt in range(candidate_count):
                img_seed = base_seed + i * candidate_count + attempt
                candidate_index = i * candidate_count + attempt
                start = time.time()
                try:
                    image = self._generate_cloud(
                        prompt, negative_prompt, width, height, steps, guidance, img_seed
                    ) if self.engine == "Cloud inference" else self._generate_local(
                        prompt, negative_prompt, width, height, steps, guidance, img_seed,
                        candidate_index, candidate_total, progress_callback,
                    )
                except Exception as exc:
                    raise GenerationError(f"AI image generation failed: {exc}") from exc

                if self.engine == "Cloud inference" and progress_callback is not None:
                    progress_callback(candidate_index, candidate_total, "Generating", 1, 1)
                if progress_callback is not None:
                    progress_callback(candidate_index, candidate_total, "Checking for stray text", 1, 1)
                text_detected, text_score, text_method = detect_stray_text(image)
                if people_requested and progress_callback is not None:
                    progress_callback(candidate_index, candidate_total, "Checking faces", 1, 1)
                face = assess_face_quality(image) if people_requested else None
                seconds = round(time.time() - start, 1)
                candidates.append({
                    "image": image,
                    "seed": img_seed,
                    "seconds": seconds,
                    "text_detected": text_detected,
                    "text_score": text_score,
                    "text_method": text_method,
                    "face": face,
                })
                if not people_requested and not text_detected:
                    break

            def candidate_rank(candidate):
                face_result = candidate["face"]
                face_available = bool(face_result and face_result.get("available"))
                face_passed = bool(face_result and face_result.get("passed"))
                face_score = float(face_result.get("score", 0)) if face_result else 0
                return (
                    not candidate["text_detected"],
                    not face_available or face_passed,
                    face_score,
                    -float(candidate["text_score"]),
                )

            best = max(candidates, key=candidate_rank)
            text_status = (
                f"stray text detected ({best['text_method']})"
                if best["text_detected"]
                else f"clear ({best['text_method']})"
            )
            face_result = best["face"]
            if face_result is None:
                face_status = "not required"
            elif not face_result.get("available"):
                face_status = "unavailable: install OpenCV face detector"
            else:
                state = "passed" if face_result.get("passed") else "failed"
                face_status = f"{state}: {face_result.get('reason', 'quality gate result unavailable')}"
            results.append(GeneratedImage(
                image=best["image"],
                seed=best["seed"],
                seconds=best["seconds"],
                text_gate_result=text_status,
                face_gate_result=face_status,
            ))

        return results

    def _generate_local(
        self, prompt, negative_prompt, width, height, steps, guidance, seed,
        candidate_index, candidate_total, callback,
    ):
        import torch

        device = self.device_info.device
        generator = torch.Generator(device="cpu" if device == "mps" else device).manual_seed(seed)
        kwargs = dict(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        if self.model_cfg["uses_negative"]:
            kwargs["negative_prompt"] = negative_prompt
        if callback is not None:
            def on_step(pipe, step, timestep, callback_kwargs):
                callback(candidate_index, candidate_total, "Generating", step + 1, steps)
                return callback_kwargs
            kwargs["callback_on_step_end"] = on_step
        with torch.inference_mode():
            return self.pipe(**kwargs).images[0]

    def _generate_cloud(self, prompt, negative_prompt, width, height, steps, guidance, seed):
        from huggingface_hub import InferenceClient

        token = _get_hf_token()
        if not token:
            raise GenerationError("Cloud inference requires HF_TOKEN.")
        errors = []
        for model in (config.HF_CLOUD_MODEL, config.HF_CLOUD_FALLBACK_MODEL):
            client = InferenceClient(model=model, token=token, timeout=180)
            for attempt in range(3):
                parameters = {
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "guidance_scale": guidance,
                    "num_inference_steps": steps,
                    "seed": seed,
                }
                try:
                    return client.text_to_image(prompt, **parameters)
                except TypeError:
                    parameters.pop("negative_prompt", None)
                    try:
                        return client.text_to_image(prompt, **parameters)
                    except Exception as exc:
                        errors.append(exc)
                        status = getattr(getattr(exc, "response", None), "status_code", None)
                        if status == 503 and attempt < 2:
                            time.sleep(2 ** attempt)
                            continue
                        break
                except Exception as exc:
                    errors.append(exc)
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if status == 503 and attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    break
        detail = str(errors[-1])[:220] if errors else "unknown provider error"
        raise GenerationError(f"Cloud models are unavailable after retries. Details: {detail}")


def unload_cuda_memory() -> None:
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
