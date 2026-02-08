"""
Vision Analyzer - GPT-4 Vision Integration
Extracts keyframes from video and analyzes them with GPT-4 Vision
to understand the concept, narrative structure, and visual style.
"""
import cv2
import base64
import os
import json
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
from PIL import Image
import io
from collections import Counter
import numpy as np


class VisionAnalyzer:
    """Analyzes video content using GPT-4 Vision to understand the trending concept"""
    
    VISION_PROMPT = """You are an expert social media content analyst specializing in viral video formats. Analyze these keyframes from a trending video with extreme attention to visual detail.

For each frame, provide DETAILED analysis:
1. **Scene Content**: 
   - What's happening (person talking, product shot, text overlay, transformation, etc.)
   - Subject positioning (center, rule of thirds, off-center)
   - Camera angle (eye-level, high angle, low angle, bird's eye, worm's eye)
   - Shot type (close-up, medium shot, wide shot, extreme close-up)
2. **Visual Style** (BE SPECIFIC):
   - **Colors**: Dominant colors, color temperature (warm/cool), saturation level, contrast
   - **Lighting**: Direction (front, side, back, rim), quality (soft/hard), intensity, shadows/highlights
   - **Composition**: Rule of thirds usage, leading lines, symmetry, depth of field (shallow/deep)
   - **Effects**: Blur, grain, vignette, color grading style, transitions
3. **Text Content**: 
   - Extract ALL text exactly as shown (OCR precisely)
   - Text position (top, bottom, center, overlay)
   - Typography style (bold, script, sans-serif, serif)
   - Text size relative to frame
   - Text color and contrast
4. **Emotion/Mood**: 
   - Primary emotion evoked
   - How visual elements contribute to this emotion

Then provide COMPREHENSIVE OVERALL analysis:
1. **Why Trending**: 
   - Specific psychological triggers (curiosity gap, social proof, relatability, etc.)
   - What makes this format engaging/viral
   - Target audience appeal
2. **Hook Technique**: 
   - Exact mechanism used in first 1-2 seconds
   - Visual hook (surprising visual, text question, transformation start)
   - Audio hook (music choice, sound effect, voice tone)
3. **Narrative Formula**: 
   - Story structure (Hook → Problem → Solution → CTA, or other pattern)
   - Pacing rhythm (fast cuts, slow build, etc.)
   - Emotional arc progression
4. **Visual Style Summary**: 
   - Overall aesthetic (minimalist, maximalist, retro, modern, etc.)
   - Consistent visual elements across frames
   - Color palette consistency
   - Lighting style consistency
5. **Key Elements to Replicate**: 
   - Critical visual elements that MUST be present
   - Composition patterns
   - Text overlay patterns
   - Motion/transition styles
6. **Technical Details**: 
   - Camera movement patterns (static, pan, zoom, handheld)
   - Editing style (jump cuts, smooth transitions, etc.)
   - Aspect ratio usage (9:16 vertical, etc.)

Output ONLY valid JSON in this exact format:
{
  "frames": [
    {
      "frame_number": 1,
      "timestamp": "0.0s",
      "scene_content": "detailed description including camera angle and shot type",
      "visual_style": "specific colors, lighting direction/quality, composition technique, effects",
      "text_content": "exact text if any, or null",
      "text_position": "top/bottom/center/overlay",
      "text_style": "typography description",
      "emotion": "curiosity/excitement/frustration/satisfaction/etc",
      "camera_angle": "eye-level/high/low/bird's-eye/worm's-eye",
      "shot_type": "close-up/medium/wide/extreme-close-up"
    }
  ],
  "overall_analysis": {
    "why_trending": "detailed explanation of viral appeal with psychological triggers",
    "hook_technique": "specific mechanism and how it grabs attention",
    "narrative_formula": "Hook → Problem → Solution → CTA",
    "visual_style_summary": "detailed aesthetic description with technical terms",
    "key_elements": ["specific element 1", "specific element 2", "specific element 3"],
    "content_type": "talking_head/product_showcase/text_story/transformation/tutorial/etc",
    "pacing": "fast/medium/slow",
    "text_usage": "heavy/moderate/minimal/none",
    "color_palette": "description of dominant colors and color scheme",
    "lighting_style": "consistent lighting approach across video",
    "composition_pattern": "recurring composition techniques used"
  }
}"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI client for GPT-4 Vision"""
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("VISION_MODEL", "gpt-4o")  # Vision-capable model
        self.temp_dir = "temp/frames"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.max_frames_default = int(os.getenv("VISION_MAX_FRAMES", "10"))
        self.temperature = float(os.getenv("VISION_TEMPERATURE", "0.2"))
        self.image_detail = os.getenv("VISION_IMAGE_DETAIL", "high")
        self.use_structured_outputs = os.getenv("VISION_STRUCTURED_OUTPUTS", "true").lower() in ("1", "true", "yes")
        
        # Store extracted keyframe paths for img2img generation
        self.keyframe_paths = []
    
    def analyze_video(self, video_path: str, max_frames: int = 8, job_id: str = None) -> Dict[str, Any]:
        """
        Main analysis function.
        Extracts keyframes and analyzes with GPT-4 Vision.
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to analyze (default 8 for cost efficiency)
            job_id: Optional job ID for unique frame naming
        
        Returns:
            Vision analysis result with frame-by-frame and overall analysis,
            including saved keyframe paths for img2img generation.
        """
        print(f"  → Extracting keyframes from video...")
        max_frames = max_frames or self.max_frames_default
        keyframes = self._extract_keyframes(video_path, max_frames, job_id)
        
        if not keyframes:
            raise ValueError("Could not extract any keyframes from video")
        
        print(f"  → Extracted {len(keyframes)} keyframes, analyzing with GPT-4 Vision...")
        analysis = self._analyze_with_vision(keyframes)
        
        # Add frame paths to the frames in analysis
        # Map the paths to the analyzed frames by index
        for i, frame in enumerate(analysis.get("frames", [])):
            if i < len(keyframes):
                frame["frame_path"] = keyframes[i].get("frame_path")
        
        # Extract color palette from keyframes
        color_palette = self._extract_color_palette(keyframes)
        
        # Add metadata including keyframe paths and color palette
        analysis["metadata"] = {
            "video_path": video_path,
            "frames_analyzed": len(keyframes),
            "frame_timestamps": [kf["timestamp"] for kf in keyframes],
            "keyframe_paths": [kf.get("frame_path") for kf in keyframes],
            "color_palette": color_palette
        }
        
        # Also add color palette to overall_analysis for easy access
        if "overall_analysis" in analysis:
            analysis["overall_analysis"]["color_palette"] = color_palette
        
        return analysis
    
    def _extract_keyframes(self, video_path: str, max_frames: int, job_id: str = None) -> List[Dict[str, Any]]:
        """
        Extract evenly-spaced keyframes from video.
        Returns list of dicts with frame data, timestamp, and saved file path.
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to extract
            job_id: Optional job ID for unique frame naming
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        if duration == 0:
            cap.release()
            raise ValueError("Video has zero duration")
        
        # Calculate frame intervals (evenly spaced)
        # Always include first frame and distribute rest evenly
        if max_frames >= total_frames:
            frame_indices = list(range(total_frames))
        else:
            # Evenly space frames across duration
            frame_indices = []
            for i in range(max_frames):
                # Distribute from start to near-end (leave some margin)
                progress = i / (max_frames - 1) if max_frames > 1 else 0
                frame_idx = int(progress * (total_frames - 1))
                frame_indices.append(frame_idx)
        
        keyframes = []
        self.keyframe_paths = []  # Reset stored paths
        
        # Generate unique prefix for this extraction
        import hashlib
        video_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
        frame_prefix = f"{job_id}_{video_hash}" if job_id else video_hash
        
        for idx, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            timestamp = frame_idx / fps
            
            # Save original frame at higher resolution for img2img
            # Save full-size frame for reference image generation
            frame_filename = f"keyframe_{frame_prefix}_{idx + 1}.jpg"
            frame_path = os.path.join(self.temp_dir, frame_filename)
            
            # Save frame at good quality for img2img reference
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            self.keyframe_paths.append(frame_path)
            
            # Resize frame for API efficiency (max 1024px on longest side)
            frame_resized = self._resize_frame(frame, max_size=1024)
            
            # Convert to base64 (for GPT-4 Vision API)
            _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            keyframes.append({
                "frame_index": idx + 1,
                "original_frame_number": frame_idx,
                "timestamp": round(timestamp, 2),
                "base64_image": base64_image,
                "frame_path": frame_path  # Include saved file path
            })
        
        cap.release()
        print(f"    ✓ Saved {len(keyframes)} keyframes to {self.temp_dir}")
        return keyframes
    
    def get_keyframe_paths(self) -> List[str]:
        """Return list of saved keyframe file paths"""
        return self.keyframe_paths
    
    def _resize_frame(self, frame, max_size: int = 1024):
        """Resize frame to fit within max_size while maintaining aspect ratio"""
        height, width = frame.shape[:2]
        
        if max(height, width) <= max_size:
            return frame
        
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    def _extract_color_palette(self, keyframes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract dominant color palette from keyframes.
        Analyzes all frames and returns a consistent color palette.
        
        Args:
            keyframes: List of keyframe dictionaries with 'frame_path' keys
            
        Returns:
            Dictionary with color palette information including:
            - description: Text description of the color scheme
            - primary_colors: List of dominant color hex codes
            - color_temperature: warm/cool/neutral
            - saturation_level: high/medium/low
        """
        if not keyframes:
            return {
                "description": "No frames available",
                "primary_colors": [],
                "color_temperature": "neutral",
                "saturation_level": "medium"
            }
        
        # Collect all pixel colors from all frames
        all_colors = []
        
        for kf in keyframes:
            frame_path = kf.get("frame_path")
            if not frame_path or not os.path.exists(frame_path):
                continue
            
            try:
                # Load image using PIL
                img = Image.open(frame_path)
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize for faster processing (max 200px on longest side)
                img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                
                # Convert to numpy array
                img_array = np.array(img)
                # Reshape to list of RGB pixels
                pixels = img_array.reshape(-1, 3)
                
                # Sample pixels (take every 10th pixel for efficiency)
                sampled_pixels = pixels[::10]
                all_colors.extend(sampled_pixels)
                
            except Exception as e:
                print(f"    ⚠ Failed to extract colors from frame {frame_path}: {e}")
                continue
        
        if not all_colors:
            return {
                "description": "Unable to extract colors",
                "primary_colors": [],
                "color_temperature": "neutral",
                "saturation_level": "medium"
            }
        
        # Convert to numpy array for processing
        all_colors = np.array(all_colors)
        
        # Use simple histogram approach (no external dependencies needed)
        # Quantize colors to reduce unique values and find dominant colors
        quantized = (all_colors // 32) * 32  # Quantize to 8 levels per channel (32-step quantization)
        
        # Count color frequencies using numpy
        # Convert to tuple for hashing, then count
        color_tuples = [tuple(c) for c in quantized]
        color_counter = Counter(color_tuples)
        
        # Get top 5 most frequent colors
        most_common = color_counter.most_common(5)
        dominant_colors = np.array([list(color) for color, count in most_common])
        
        # Convert to hex codes (top 3)
        primary_colors = [f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}" for c in dominant_colors[:3]]
        
        # Analyze color temperature (warm vs cool)
        # Warm colors have more red/yellow, cool colors have more blue
        avg_r = np.mean(dominant_colors[:, 0])
        avg_b = np.mean(dominant_colors[:, 2])
        
        if avg_r > avg_b + 30:
            color_temp = "warm"
        elif avg_b > avg_r + 30:
            color_temp = "cool"
        else:
            color_temp = "neutral"
        
        # Analyze saturation
        # Calculate average saturation (distance from gray)
        saturations = []
        for color in dominant_colors:
            r, g, b = color
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            if max_val == 0:
                sat = 0
            else:
                sat = (max_val - min_val) / max_val
            saturations.append(sat)
        
        avg_saturation = np.mean(saturations)
        if avg_saturation > 0.5:
            sat_level = "high"
        elif avg_saturation > 0.25:
            sat_level = "medium"
        else:
            sat_level = "low"
        
        # Create description
        color_names = []
        for color in dominant_colors[:3]:
            r, g, b = color
            # Simple color name approximation
            if r > g and r > b:
                color_names.append("reddish")
            elif g > r and g > b:
                color_names.append("greenish")
            elif b > r and b > g:
                color_names.append("bluish")
            elif r > 200 and g > 200:
                color_names.append("yellowish")
            elif r > 150 and g > 150 and b > 150:
                color_names.append("light")
            elif r < 50 and g < 50 and b < 50:
                color_names.append("dark")
            else:
                color_names.append("neutral")
        
        description = f"{color_temp.capitalize()} color palette with {sat_level} saturation, dominated by {', '.join(color_names[:2])} tones"
        
        return {
            "description": description,
            "primary_colors": primary_colors,
            "color_temperature": color_temp,
            "saturation_level": sat_level
        }
    
    def _analyze_with_vision(self, keyframes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Send keyframes to GPT-4 Vision for analysis.
        """
        # Build the message content with images
        content = [
            {
                "type": "text",
                "text": f"{self.VISION_PROMPT}\n\nI'm sending you {len(keyframes)} keyframes from a trending video. Analyze them in order."
            }
        ]
        
        # Add each frame as an image
        for kf in keyframes:
            content.append({
                "type": "text",
                "text": f"\n--- Frame {kf['frame_index']} (timestamp: {kf['timestamp']}s) ---"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{kf['base64_image']}",
                    "detail": self.image_detail  # Use high detail for better text recognition
                }
            })
        
        # Call GPT-4 Vision
        try:
            response = None
            if self.use_structured_outputs:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=4000,
                    temperature=self.temperature,
                    response_format={"type": "json_schema", "json_schema": self._get_vision_schema()}
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=4000,
                    temperature=self.temperature
                )
            
            response_text = response.choices[0].message.content
            
            # Try to parse JSON from response
            # Handle potential markdown code blocks
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            analysis = json.loads(response_text)
            return analysis
            
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse Vision API response as JSON: {e}")
            # Return a basic structure with the raw response
            return {
                "frames": [],
                "overall_analysis": {
                    "why_trending": "Analysis parsing failed",
                    "hook_technique": "Unknown",
                    "narrative_formula": "Unknown",
                    "visual_style_summary": "Unknown",
                    "key_elements": [],
                    "content_type": "unknown",
                    "pacing": "medium",
                    "text_usage": "unknown",
                    "raw_response": response_text
                }
            }
        except Exception as e:
            print(f"Vision API error: {e}")
            raise

    def _get_vision_schema(self) -> Dict[str, Any]:
        """
        JSON schema for structured vision output.
        """
        return {
            "name": "vision_analysis",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "frames": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "frame_number": {"type": "integer"},
                                "timestamp": {"type": "string"},
                                "scene_content": {"type": "string"},
                                "visual_style": {"type": "string"},
                                "text_content": {"type": ["string", "null"]},
                                "text_position": {"type": "string"},
                                "text_style": {"type": "string"},
                                "emotion": {"type": "string"},
                                "camera_angle": {"type": "string"},
                                "shot_type": {"type": "string"}
                            },
                            "required": [
                                "frame_number", "timestamp", "scene_content", "visual_style",
                                "text_content", "text_position", "text_style", "emotion",
                                "camera_angle", "shot_type"
                            ]
                        }
                    },
                    "overall_analysis": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "why_trending": {"type": "string"},
                            "hook_technique": {"type": "string"},
                            "narrative_formula": {"type": "string"},
                            "visual_style_summary": {"type": "string"},
                            "key_elements": {"type": "array", "items": {"type": "string"}},
                            "content_type": {"type": "string"},
                            "pacing": {"type": "string"},
                            "text_usage": {"type": "string"},
                            "color_palette": {"type": "string"},
                            "lighting_style": {"type": "string"},
                            "composition_pattern": {"type": "string"}
                        },
                        "required": [
                            "why_trending", "hook_technique", "narrative_formula",
                            "visual_style_summary", "key_elements", "content_type",
                            "pacing", "text_usage", "color_palette", "lighting_style",
                            "composition_pattern"
                        ]
                    }
                },
                "required": ["frames", "overall_analysis"]
            }
        }
    
    def extract_text_from_frames(self, video_path: str, timestamps: List[float] = None) -> List[Dict[str, Any]]:
        """
        Extract frames at specific timestamps for OCR analysis.
        Useful for getting exact text content at key moments.
        
        Args:
            video_path: Path to video
            timestamps: List of timestamps (in seconds) to extract. If None, extracts every 2 seconds.
        
        Returns:
            List of text extractions with timestamps
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        if timestamps is None:
            # Default: every 2 seconds
            timestamps = [i * 2.0 for i in range(int(duration / 2) + 1)]
        
        text_results = []
        
        for ts in timestamps:
            frame_idx = int(ts * fps)
            if frame_idx >= total_frames:
                continue
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            frame = self._resize_frame(frame, max_size=1024)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            # Quick OCR with GPT-4 Vision
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Extract ALL text visible in this image. Return ONLY the text, nothing else. If no text, return 'NO_TEXT'."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=500,
                    temperature=0
                )
                
                text = response.choices[0].message.content.strip()
                if text != "NO_TEXT":
                    text_results.append({
                        "timestamp": ts,
                        "text": text
                    })
            except Exception as e:
                print(f"OCR failed at {ts}s: {e}")
                continue
        
        cap.release()
        return text_results
    
    def save_analysis(self, analysis: Dict[str, Any], output_path: str):
        """Save vision analysis to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)


if __name__ == "__main__":
    # Test the vision analyzer
    analyzer = VisionAnalyzer()
    # analysis = analyzer.analyze_video("test_video.mp4")
    # print(json.dumps(analysis, indent=2))
