"""
LLM Creative Director - Enhanced with Full Concept Context
Takes the complete Concept Blueprint and generates a product-adapted Ad Blueprint.
Now understands WHY a video is trending and replicates the concept, not just timing.
"""
import json
from typing import Dict, Any, Optional, List
from openai import OpenAI
import os


class LLMDirector:
    """
    Enhanced Creative Director that uses full concept understanding
    to generate trend-replicating ad blueprints.
    """
    
    SYSTEM_PROMPT = """You are an expert social media ad creative director who specializes in replicating trending video formats with technical precision.

Your job is to take a CONCEPT BLUEPRINT (which describes WHY a video is trending, its narrative structure, visual style, and key elements) and create a NEW AD BLUEPRINT that:
1. REPLICATES the trending concept/formula (the "why it works")
2. ADAPTS it for a new product
3. MAINTAINS the same narrative structure, pacing, and emotional journey
4. CREATES original visuals and text that follow the same STYLE

CRITICAL RULES FOR VISUAL PROMPTS:
- Use TECHNICAL PHOTOGRAPHY TERMS: camera angles (eye-level, high-angle, low-angle, bird's-eye), shot types (close-up, medium shot, wide shot), lighting (soft front lighting, dramatic side lighting, rim lighting), depth of field (shallow focus, deep focus)
- Include COMPOSITION DETAILS: rule of thirds, leading lines, symmetry, negative space usage
- Specify COLOR DETAILS: dominant colors, color temperature (warm 3000K, cool 6500K), saturation level, contrast
- Describe MOOD/ATMOSPHERE: cinematic, commercial, documentary-style, lifestyle, aspirational
- Be SPECIFIC about PRODUCT PLACEMENT: center frame, rule of thirds intersection, foreground/background
- Include ENVIRONMENTAL CONTEXT: background style, props, setting atmosphere

CRITICAL RULES FOR TEXT OVERLAYS:
- Match the PATTERN (question format, statement format, emoji usage, etc.)
- Match the POSITION (top, bottom, center overlay)
- Match the STYLE (bold sans-serif, script font, etc.)
- Match the TIMING (when text appears relative to visual)
- Never copy exact text, but replicate the FORMAT and STYLE

CRITICAL RULES:
- Output ONLY valid JSON
- Match the segment structure, roles, and timing from the concept blueprint EXACTLY
- Replicate the HOOK TECHNIQUE that makes the original engaging
- Follow the same NARRATIVE FORMULA (e.g., Hook → Problem → Solution → CTA)
- Match the VISUAL STYLE described for each segment with technical precision
- Create text overlays that follow the same PATTERN as the original
- Consider the AUDIO TYPE (speech_over_music, music_only, etc.) when creating content
- Use SYNC POINTS for impactful moments in your visual prompts
- Visual prompts MUST be detailed enough for AI image generation (include camera angle, lighting, composition, mood)"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI client"""
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("LLM_MODEL", "gpt-4o")
        # Use Structured Outputs when available for higher JSON quality.
        self.use_structured_outputs = os.getenv("LLM_STRUCTURED_OUTPUTS", "true").lower() in ("1", "true", "yes")

    def _get_ad_blueprint_schema(self) -> Dict[str, Any]:
        """
        JSON schema for Structured Outputs. This improves response quality
        by enforcing required fields and types.
        """
        return {
            "name": "ad_blueprint",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "trend_replication": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "original_hook": {"type": "string"},
                            "adapted_hook": {"type": "string"},
                            "narrative_preserved": {"type": "boolean"},
                            "style_matched": {"type": "boolean"}
                        },
                        "required": ["original_hook", "adapted_hook", "narrative_preserved", "style_matched"]
                    },
                    "segments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "id": {"type": "integer"},
                                "role": {"type": "string"},
                                "duration": {"type": "number"},
                                "start": {"type": "number"},
                                "end": {"type": "number"},
                                "visual_prompt": {"type": "string"},
                                "overlay_text": {"type": ["string", "null"]},
                                "visual_style": {"type": "string"},
                                "emotion": {"type": "string"},
                                "motion": {"type": "string"},
                                "sync_to_beat": {"type": "boolean"}
                            },
                            "required": [
                                "id", "role", "duration", "start", "end", "visual_prompt",
                                "overlay_text", "visual_style", "emotion", "motion", "sync_to_beat"
                            ]
                        }
                    },
                    "cta": {"type": "string"},
                    "overall_style": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "color_palette": {"type": "string"},
                            "text_style": {"type": "string"},
                            "transitions": {"type": "string"}
                        },
                        "required": ["color_palette", "text_style", "transitions"]
                    }
                },
                "required": ["trend_replication", "segments", "cta", "overall_style"]
            }
        }
    
    def generate_ad_blueprint(
        self,
        concept_blueprint: Dict[str, Any],
        product_description: str,
        brand_tone: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate ad blueprint from concept blueprint and product description.
        Returns enhanced Ad Blueprint JSON.
        """
        # Build the prompt with full context
        user_prompt = self._build_enhanced_prompt(
            concept_blueprint,
            product_description,
            brand_tone
        )
        
        # Call LLM
        response = None
        try:
            if self.use_structured_outputs:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.5,
                    response_format={"type": "json_schema", "json_schema": self._get_ad_blueprint_schema()}
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )
        except Exception as e:
            # Fallback to JSON mode if structured outputs fail
            print(f"  ⚠ Structured outputs failed: {e}. Falling back to JSON mode.")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
        
        # Parse JSON response
        try:
            ad_blueprint = json.loads(response.choices[0].message.content)
            
            # Ensure required fields exist
            ad_blueprint = self._validate_and_enhance_blueprint(
                ad_blueprint,
                concept_blueprint
            )
            
            return ad_blueprint
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}")
    
    def _build_enhanced_prompt(
        self,
        concept_blueprint: Dict[str, Any],
        product_description: str,
        brand_tone: Optional[str]
    ) -> str:
        """Build comprehensive prompt with full concept context"""
        
        # Extract trend analysis
        trend = concept_blueprint.get("trend_analysis", {})
        segments = concept_blueprint.get("segments", [])
        audio = concept_blueprint.get("audio", {})
        
        # Build the prompt
        prompt = f"""Create an ad blueprint for this product that REPLICATES the trending video concept:

═══════════════════════════════════════════════════════════════════
PRODUCT TO ADVERTISE:
═══════════════════════════════════════════════════════════════════
{product_description}
"""
        
        if brand_tone:
            prompt += f"""
BRAND TONE:
{brand_tone}
"""
        
        prompt += f"""
═══════════════════════════════════════════════════════════════════
WHY THE ORIGINAL VIDEO IS TRENDING:
═══════════════════════════════════════════════════════════════════
• Viral Appeal: {trend.get('why_trending', 'Unknown')}
• Hook Technique: {trend.get('hook_technique', 'Unknown')}
• Narrative Formula: {trend.get('narrative_formula', 'Unknown')}
• Visual Style: {trend.get('visual_style', 'Unknown')}
• Content Type: {trend.get('content_type', 'Unknown')}
• Pacing: {trend.get('pacing', 'medium')}
• Text Usage: {trend.get('text_usage', 'moderate')}
• Key Elements to Replicate: {json.dumps(trend.get('key_elements', []))}

═══════════════════════════════════════════════════════════════════
AUDIO INFORMATION:
═══════════════════════════════════════════════════════════════════
• Audio Type: {audio.get('audio_type', 'music_only')}
• BPM: {audio.get('bpm', 120)}
• Has Speech: {audio.get('has_speech', False)}
• Original Transcript: {audio.get('transcript', 'N/A') or 'N/A'}
• Sync Points (important moments): {json.dumps(audio.get('sync_points', []))}

═══════════════════════════════════════════════════════════════════
SEGMENT-BY-SEGMENT BREAKDOWN (REPLICATE THIS STRUCTURE):
═══════════════════════════════════════════════════════════════════
"""
        
        for segment in segments:
            prompt += f"""
--- Segment {segment.get('id', '?')} ({segment.get('role', 'UNKNOWN')}) ---
• Duration: {segment.get('duration', 1)}s (from {segment.get('start', 0)}s to {segment.get('end', 1)}s)
• Original Scene: {segment.get('scene_content', 'Unknown')}
• Visual Style: {segment.get('visual_style', 'Standard')}
• Original Text: {segment.get('text_content', 'None')}
• Emotion to Evoke: {segment.get('emotion', 'neutral')}
• Beat Sync: {segment.get('has_beat_sync', False)}
"""
        
        prompt += f"""
═══════════════════════════════════════════════════════════════════
GENERATE AD BLUEPRINT:
═══════════════════════════════════════════════════════════════════
Create a JSON with this EXACT structure. Each segment should:
- Keep the SAME duration and role
- Create a NEW visual_prompt that advertises the product but follows the SAME STYLE
- Create NEW overlay_text that matches the PATTERN of the original
- Match the emotion and visual style
- Note if it should sync to a beat

{{
  "trend_replication": {{
    "original_hook": "description of original hook",
    "adapted_hook": "how you're adapting it for the product",
    "narrative_preserved": true/false,
    "style_matched": true/false
  }},
  "segments": [
    {{
      "id": 1,
      "role": "HOOK/PROBLEM/BUILDUP/SOLUTION/RESULT/CTA",
      "duration": <exact duration from concept blueprint>,
      "start": <start time>,
      "end": <end time>,
      "visual_prompt": "<detailed description of what to show - product-focused but matching the original STYLE>",
      "overlay_text": "<text to show, matching the PATTERN of original, or null>",
      "visual_style": "<style description matching original>",
      "emotion": "<emotion to evoke>",
      "motion": "static/slow_pan/fast_zoom/handheld",
      "sync_to_beat": true/false
    }}
  ],
  "cta": "<call-to-action text>",
  "overall_style": {{
    "color_palette": "description of colors to use",
    "text_style": "description of text style (bold, subtle, etc)",
    "transitions": "description of transition style"
  }}
}}

IMPORTANT:
- The number of segments MUST match the concept blueprint ({len(segments)} segments)
- Each segment's duration MUST match the original
- The visual_prompt should be DETAILED enough for AI image generation
- If the original had text overlays, your version should too (but with product-relevant text)
- Match the emotional journey: {' → '.join([s.get('emotion', 'neutral') for s in segments])}
"""
        
        return prompt
    
    def _validate_and_enhance_blueprint(
        self,
        ad_blueprint: Dict[str, Any],
        concept_blueprint: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and enhance the generated blueprint.
        Ensures all required fields exist and segments match.
        Copies reference_frame_path from concept segments for img2img generation.
        """
        # Ensure segments exist
        if "segments" not in ad_blueprint:
            ad_blueprint["segments"] = []
        
        # Ensure we have the right number of segments
        concept_segments = concept_blueprint.get("segments", [])
        generated_segments = ad_blueprint.get("segments", [])
        
        # If segment count doesn't match, adjust
        if len(generated_segments) != len(concept_segments):
            print(f"  ⚠ Segment count mismatch: expected {len(concept_segments)}, got {len(generated_segments)}")
            # Adjust segments to match
            ad_blueprint["segments"] = self._adjust_segments(
                generated_segments,
                concept_segments
            )
        
        # Ensure each segment has required fields and copy reference frame paths
        for i, segment in enumerate(ad_blueprint["segments"]):
            if "duration" not in segment and i < len(concept_segments):
                segment["duration"] = concept_segments[i].get("duration", 2.0)
            if "visual_prompt" not in segment:
                segment["visual_prompt"] = "Product showcase"
            if "motion" not in segment:
                segment["motion"] = "static"
            if "role" not in segment and i < len(concept_segments):
                segment["role"] = concept_segments[i].get("role", "CONTENT")
            
            # Copy reference_frame_path from concept blueprint for img2img generation
            if i < len(concept_segments):
                segment["reference_frame_path"] = concept_segments[i].get("reference_frame_path")
        
        # Ensure CTA exists
        if "cta" not in ad_blueprint:
            ad_blueprint["cta"] = "Check it out!"
        
        # Add legacy "shots" field for backward compatibility
        ad_blueprint["shots"] = self._segments_to_shots(ad_blueprint["segments"])
        
        return ad_blueprint
    
    def _adjust_segments(
        self,
        generated: List[Dict[str, Any]],
        concept: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Adjust generated segments to match concept segment count"""
        result = []
        
        for i, concept_seg in enumerate(concept):
            if i < len(generated):
                # Use generated segment but ensure timing matches
                seg = generated[i].copy()
                seg["duration"] = concept_seg.get("duration", 2.0)
                seg["start"] = concept_seg.get("start", 0)
                seg["end"] = concept_seg.get("end", 2.0)
                seg["role"] = concept_seg.get("role", "CONTENT")
                # Copy reference frame path for img2img
                seg["reference_frame_path"] = concept_seg.get("reference_frame_path")
            else:
                # Create placeholder segment
                seg = {
                    "id": i + 1,
                    "role": concept_seg.get("role", "CONTENT"),
                    "duration": concept_seg.get("duration", 2.0),
                    "start": concept_seg.get("start", 0),
                    "end": concept_seg.get("end", 2.0),
                    "visual_prompt": f"Product showcase - {concept_seg.get('role', 'scene')}",
                    "overlay_text": None,
                    "visual_style": concept_seg.get("visual_style", "Standard"),
                    "emotion": concept_seg.get("emotion", "neutral"),
                    "motion": "static",
                    "sync_to_beat": concept_seg.get("has_beat_sync", False),
                    # Copy reference frame path for img2img
                    "reference_frame_path": concept_seg.get("reference_frame_path")
                }
            result.append(seg)
        
        return result
    
    def _segments_to_shots(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert segments to legacy shots format for backward compatibility"""
        shots = []
        for segment in segments:
            shot = {
                "duration": segment.get("duration", 2.0),
                "visual_prompt": segment.get("visual_prompt", "Product showcase"),
                "overlay_text": segment.get("overlay_text"),
                "motion": segment.get("motion", "static"),
                "energy": self._role_to_energy(segment.get("role", "CONTENT"))
            }
            shots.append(shot)
        return shots
    
    def _role_to_energy(self, role: str) -> str:
        """Map role to energy level"""
        high_energy = ["HOOK", "CTA", "SOLUTION"]
        low_energy = ["PROBLEM", "BUILDUP"]
        
        if role in high_energy:
            return "high"
        elif role in low_energy:
            return "low"
        else:
            return "medium"
    
    def save_blueprint(self, blueprint: Dict[str, Any], output_path: str):
        """Save ad blueprint to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(blueprint, f, indent=2)


if __name__ == "__main__":
    # Test with sample data
    director = LLMDirector()
    # concept_bp = {...}
    # ad_bp = director.generate_ad_blueprint(concept_bp, "A magnetic cable organizer", "playful")
    # print(json.dumps(ad_bp, indent=2))
