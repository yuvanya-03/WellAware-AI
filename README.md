---
title: WellAware AI
sdk: docker
app_port: 7860
---

# WellAware AI

Patient Wellness Campaign Visual Generator for non-clinical, patient-facing
wellness awareness. It creates an illustration first, then adds editable copy
using Pillow. It is a draft-design tool, not a medical communications approval
system.

> **Required footer:** AI-generated, non-clinical wellness imagery. Requires
> internal medical/legal/regulatory (MLR) review before public use.

## Architecture

- `app.py` provides a three-step Streamlit workflow, preview tabs, downloads,
  a combined poster gallery containing user-created and persistent preset
  campaigns, and audit-log download.
- `modules/prompt_processor.py` validates input, removes lettering-oriented
  terms from image prompts, lints copy, and uses the CLIP tokenizer.
- `modules/image_generator.py` runs local SDXL (a GPU is recommended), with
  SD 1.5 and SD-Turbo draft options. Hugging Face Spaces select their hosted
  image service automatically; it is not shown as a local-app control.
- `modules/image_utils.py` checks artwork for stray text, optionally scores
  faces with OpenCV, composes the poster, and saves PNG/JPG data.

## Run locally

Use Python 3.11 or 3.12. Install PyTorch using the command appropriate for your
GPU from <https://pytorch.org/get-started/locally/> before installing the rest.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/download_fonts.py
Copy-Item .env.example .env
streamlit run app.py
```

The local app uses the selected local model. SDXL is intended for a GPU; CPU
generation can take a long time. Model weights are downloaded on first use and
require substantial disk space and GPU memory.

Open **Gallery** and choose **Generate missing preset campaigns** to create the
preset campaign set with SDXL. Each completed poster is saved under
`generated_images/preset_campaigns` and appears in the same gallery as posters
you create. Existing app-generated PNGs are also restored into the gallery on
startup. If generation is interrupted, use the button again to continue. The
sidebar shows one total poster count.

Run focused tests with:

```powershell
python -m pytest -q
```

## Render deployment (free tier)

The root `render.yaml` provisions a Docker web service on Render's free plan.
The Docker command binds to Render's `PORT`, and `SPACE_ID=render` selects the
hosted Hugging Face inference engine. The `HF_TOKEN` value is requested
separately by Render and must not be committed to the repository.

1. Push this project to a GitHub repository, then connect it from the Render
   Dashboard using **New → Blueprint**.
2. Select the repository containing `render.yaml` and deploy the Blueprint.
3. Add a Hugging Face token with Inference Provider access as the Render
   service's `HF_TOKEN` secret, then redeploy.
4. Open the public `*.onrender.com` URL shown on the service page.

Render's free service sleeps after inactivity and has an ephemeral filesystem:
generated posters and the gallery can disappear after sleep, restart, or
redeploy. Hosted inference is subject to Hugging Face availability and quota.

## Hugging Face Spaces deployment

The Docker image uses lightweight hosted requirements. On Spaces, the app
selects the hosted Hugging Face image service automatically (FLUX.1-schnell,
then SDXL fallback). Provider availability and quotas apply. The local GUI does
not show a hosted/cloud engine selector.

### Click-by-click

1. Sign in at <https://huggingface.co> and select **New → Space**.
2. Enter a Space name, choose **Docker** as the SDK, choose **Public**, then
   select **Create Space**.
3. Open **Settings → Variables and secrets → New secret**. Add `HF_TOKEN` with a
   Hugging Face token that can access the hosted image service.
4. On your computer, open PowerShell in this project folder, authenticate, and
   push the files using the commands below. Replace both placeholders with the
   Space owner and name.
5. Wait for the Docker build to complete, then open
   `https://<SPACE_OWNER>-<SPACE_NAME>.hf.space`. The first inference may show
   a message that the hosted image model is waking up.

```powershell
hf auth login
git init
git add .
git commit -m "Deploy WellAware AI"
git branch -M main
git remote add space https://huggingface.co/spaces/<SPACE_OWNER>/<SPACE_NAME>
git push -u space main
```

This checkout has no Git repository or Hugging Face credentials configured, so
no Space has been created or live generation verified from this workspace.

## Google Colab

Open `WellAware_AI_Colab.ipynb`, select a T4 GPU, upload the project archive,
then run the cells in order. The requirements install includes the SDXL and
cloud components; Tesseract OCR is optional and the app has a visual fallback.
The notebook starts Streamlit and Cloudflare Tunnel for temporary demos. Keep
the notebook session active while using that temporary link.

## Safety, ethics, and limitations

- Artwork prompts are sanitized and use a negative prompt where supported.
  OCR is attempted when Tesseract is available; otherwise a conservative
  edge-density heuristic is used. Neither method guarantees zero artifacts.
- Person prompts request one or two people. OpenCV Haar detection is a
  lightweight quality signal, not a reliable identity or anatomical assessment.
  Detector-unavailable status is shown rather than treated as a pass.
- Poster copy is editable and claim-like wording is flagged before export.
  Review all copy and visuals; this software does not provide medical advice,
  evidence review, accessibility certification, or MLR approval.
- Session gallery is in memory. `auditlog.csv` is local and can contain user
  prompts; treat it as potentially sensitive and do not commit it.
- Hosted inference can be unavailable, rate-limited, or subject to provider
  terms. A generated image may still contain residual text or imperfect faces
  and hands. Every exported poster requires human review.

## Secrets

Copy `.env.example` to `.env` for local work and fill secrets locally. The
`.env` file and `auditlog.csv` are ignored by Git. Never commit tokens.
