# Viral Reel Recreator (Style-Based, Not Copying)

## Goal
Build a system that:
1. Takes a **trending Instagram Reel (MP4 upload for MVP)**
2. Takes a **product description (text + optional images)**
3. Extracts the **structural style** of the reel (NOT content)
4. Uses an LLM to generate a **new ad blueprint** for the product
5. Programmatically renders a **new short video** following that blueprint

IMPORTANT:
- Do NOT pass raw video to the LLM
- Do NOT copy actors, text, scenes, or visuals
- Only reuse *structure, timing, pacing, and motion patterns*

---

## Tech Stack (Mandatory)
- Language: Python 3.10+
- Backend: FastAPI
- Video Processing: OpenCV, FFmpeg, librosa
- LLM API: OpenAI (GPT-4o or equivalent)
- Video Rendering: MoviePy or FFmpeg
- Data Format Between Steps: JSON only

---

## System Architecture

Input Video (MP4)
→ Style Extraction (CV + Audio)
→ Style Blueprint JSON
→ LLM Creative Expansion
→ Ad Blueprint JSON
→ Video Renderer
→ Final Reel (MP4, 9:16)

---

## STEP 1: API Endpoints

### POST /upload
Inputs:
- video_file (mp4)
- product_description (string)
- brand_tone (optional string)

Output:
- job_id

---

## STEP 2: Video Style Extraction (NO LLM)

Create module: style_extractor.py

### Responsibilities
Extract ONLY objective signals:

Visual:
- Shot boundaries
- Shot durations
- Camera motion (static / pan / zoom)
- Presence of text overlays (yes/no)
- Framing approximation (close / medium / wide)

Audio:
- BPM
- Beat drop timestamps
- Silence vs impact moments

---

### Output: Style Blueprint JSON

{
  "total_duration": 7.1,
  "hook_duration": 1.0,
  "shots": [
    {
      "start": 0.0,
      "end": 1.0,
      "duration": 1.0,
      "motion": "fast_zoom",
      "text_overlay": true,
      "energy": "high"
    },
    {
      "start": 1.0,
      "end": 2.5,
      "duration": 1.5,
      "motion": "handheld",
      "text_overlay": true,
      "energy": "high"
    },
    {
      "start": 2.5,
      "end": 5.0,
      "duration": 2.5,
      "motion": "static",
      "text_overlay": false,
      "energy": "medium"
    },
    {
      "start": 5.0,
      "end": 7.1,
      "duration": 2.1,
      "motion": "slow_pan",
      "text_overlay": true,
      "energy": "high"
    }
  ],
  "music": {
    "bpm": 120,
    "beat_drop": 1.1
  }
}

---

## STEP 3: LLM Prompting (Creative Director Role)

Create module: llm_director.py

System Prompt:
You are a senior performance ad creative director.
You NEVER copy visuals, text, actors, or scenes.
You ONLY adapt timing, pacing, and structure.

Inputs:
- style_blueprint.json
- product_description
- brand_tone (optional)

---

### LLM Output: Ad Blueprint JSON

{
  "shots": [
    {
      "duration": 1.0,
      "visual_prompt": "close-up of messy desk cables",
      "overlay_text": "This annoying?",
      "motion": "fast_zoom",
      "energy": "high"
    },
    {
      "duration": 1.5,
      "visual_prompt": "person discovering a magnetic cable organizer",
      "overlay_text": "There’s a fix",
      "motion": "handheld",
      "energy": "high"
    },
    {
      "duration": 2.5,
      "visual_prompt": "clean desk with organized cables",
      "overlay_text": null,
      "motion": "static",
      "energy": "medium"
    },
    {
      "duration": 2.1,
      "visual_prompt": "product close-up with satisfying click",
      "overlay_text": "Clean desk in seconds",
      "motion": "slow_pan",
      "energy": "high"
    }
  ],
  "cta": "Get yours today"
}

STRICT RULES:
- JSON only
- No prose
- No markdown
- No explanations

---

## STEP 4: Asset Generation (MVP)

Create module: asset_generator.py

- Generate one image per shot
- Animate using pan / zoom / parallax
- Placeholder images acceptable for MVP

---

## STEP 5: Video Rendering Engine

Create module: video_renderer.py

Responsibilities:
- Create 9:16 canvas (1080x1920)
- Apply motion presets per shot
- Animate overlay text
- Add background music
- Sync cuts to BPM

Output:
- final_video.mp4

---

## STEP 6: Orchestration

Create pipeline.py

Steps:
1. Extract style blueprint
2. Call LLM
3. Generate assets
4. Render video
5. Save output
6. Return video URL

---

## STEP 7: Guardrails

- Never reuse exact text or visuals
- Add ±10% jitter to timing
- Log all JSON + outputs

---

## MVP Definition
- Input: MP4 + product text
- Output: New reel under 10 seconds
- Style matches pacing
- Content is original
- No raw video sent to LLM

---

CORE PRINCIPLE:
LLM reasons about structure.
Code handles pixels.
Never mix the two.
