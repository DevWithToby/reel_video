"""
Asset Generator - Enhanced Multi-Segment Generation
Generates unique images for each segment with style consistency.
Uses Hugging Face Stable Diffusion XL for AI image generation.
"""
from typing import Dict, List, Any, Union, Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import os
import io
import requests
import hashlib
from moviepy.editor import ImageClip, CompositeVideoClip, TextClip, concatenate_videoclips, ColorClip, VideoClip


class AssetGenerator:
    """
    Enhanced asset generator that creates unique visuals for each segment
    while maintaining style consistency across the video.
    Supports img2img generation using reference frames from trending videos.
    """
    
    def __init__(self, output_dir: str = "temp/assets"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.canvas_size = (1080, 1920)  # 9:16 aspect ratio
        
        # Hugging Face API configuration
        self.hf_api_key = os.environ.get("HUGGINGFACE_API_KEY")
        self.hf_model_url = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
        # Img2Img endpoint for SDXL
        self.hf_img2img_url = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-refiner-1.0"
        self.use_ai_generation = bool(self.hf_api_key)
        # Quality controls (env-configurable)
        self.sdxl_width = int(os.environ.get("SDXL_WIDTH", "1024"))
        self.sdxl_height = int(os.environ.get("SDXL_HEIGHT", "1792"))  # 9:16-ish
        self.sdxl_steps = int(os.environ.get("SDXL_STEPS", "40"))
        self.sdxl_guidance = float(os.environ.get("SDXL_GUIDANCE", "7.0"))
        self.sdxl_seed = os.environ.get("SDXL_SEED")
        self.negative_prompt = os.environ.get(
            "SDXL_NEGATIVE_PROMPT",
            "blurry, low quality, distorted, ugly, bad anatomy, watermark, text overlay, "
            "logo, amateur, poorly lit, oversaturated, underexposed, overexposed, bad composition, "
            "cluttered, unprofessional, low resolution, pixelated, artifacts, noise, grainy, "
            "out of focus, double exposure, bad lighting, harsh shadows, unnatural colors, "
            "cartoon style, illustration, painting, sketch, abstract art, unrealistic proportions, "
            "deformed, mutated, extra limbs, missing limbs, bad hands, bad fingers, bad eyes, "
            "bad face, duplicate elements, repetitive patterns, signature, copyright, text, letters, numbers"
        )
        
        if self.use_ai_generation:
            print("✓ Hugging Face API key found - AI image generation enabled (text2img + img2img)")
        else:
            print("⚠ No Hugging Face API key - using placeholder images")
        
        # Style consistency tracking
        self.current_style = None
        self.generated_images = []
        
        # Product images for accurate product representation
        self.product_image_paths = []
        
        # Img2img settings
        self.img2img_strength = 0.6  # How much to deviate from reference (0.4-0.7 recommended)

        # Optional depth-based parallax animation (no external API cost)
        self.enable_depth_parallax = os.environ.get("ENABLE_DEPTH_PARALLAX", "false").lower() in ("1", "true", "yes")
        self.parallax_quality = os.environ.get("PARALLAX_QUALITY", "").strip().lower()
        self.midas_model = os.environ.get("MIDAS_MODEL", "midas_v21_small_256")
        self.parallax_amplitude = float(os.environ.get("PARALLAX_AMPLITUDE", "12"))
        self.parallax_vertical = float(os.environ.get("PARALLAX_VERTICAL", "6"))
        self._midas = None
        self._midas_transform = None
        self._midas_device = None
        self._apply_parallax_quality_defaults()
        self.motion_overscan = float(os.environ.get("MOTION_OVERSCAN", "1.05"))
    
    def generate_assets(
        self,
        ad_blueprint: Dict[str, Any],
        job_id: str,
        product_image_paths: List[str] = None
    ) -> List[str]:
        """
        Generate animated video sequences for each segment.
        Returns list of video file paths.
        """
        video_paths = []
        
        # Extract overall style for consistency
        self.current_style = ad_blueprint.get("overall_style", {})
        self.generated_images = []
        self.product_image_paths = product_image_paths or []
        
        # Get segments (use new format if available, fallback to shots)
        segments = ad_blueprint.get("segments", ad_blueprint.get("shots", []))
        
        print(f"  → Generating assets for {len(segments)} segments...")
        
        for i, segment in enumerate(segments):
            print(f"  → Segment {i + 1}/{len(segments)}: {segment.get('role', 'CONTENT')}")
            video_path = self._generate_animated_segment(segment, job_id, i)
            video_paths.append(video_path)
        
        return video_paths
    
    def _generate_animated_segment(
        self,
        segment: Dict[str, Any],
        job_id: str,
        segment_index: int
    ) -> str:
        """
        Generate an animated video sequence for a segment.
        Uses reference frames from trending video for img2img generation.
        """
        duration = segment.get("duration", 2.0)
        motion = segment.get("motion", "static")
        visual_prompt = segment.get("visual_prompt", "Product showcase")
        overlay_text = segment.get("overlay_text")
        role = segment.get("role", "CONTENT")
        visual_style = segment.get("visual_style", "Standard")
        emotion = segment.get("emotion", "neutral")
        reference_frame_path = segment.get("reference_frame_path")  # For img2img
        
        # Smart product image selection: Match product images to segment roles
        selected_product_image = self._select_product_image_for_segment(role, segment_index)
        
        # Step 1: Generate product image with context (using img2img if reference available)
        product_image_path = self._generate_segment_image(
            visual_prompt=visual_prompt,
            role=role,
            visual_style=visual_style,
            emotion=emotion,
            job_id=job_id,
            segment_index=segment_index,
            reference_frame_path=reference_frame_path,  # Trending video style reference
            product_image_path=selected_product_image  # Product accuracy reference
        )
        
        # Step 2: Create animated video from image
        animated_clip = self._animate_segment(
            image_path=product_image_path,
            motion=motion,
            duration=duration,
            overlay_text=overlay_text,
            role=role,
            emotion=emotion
        )
        
        # Step 3: Save animated sequence
        filename = f"{job_id}_segment_{segment_index}.mp4"
        filepath = os.path.join(self.output_dir, filename)
        
        animated_clip.write_videofile(
            filepath,
            fps=30,
            codec='libx264',
            audio_codec='aac',
            preset='medium',
            verbose=False,
            logger=None
        )
        animated_clip.close()
        
        return filepath
    
    def _generate_segment_image(
        self,
        visual_prompt: str,
        role: str,
        visual_style: str,
        emotion: str,
        job_id: str,
        segment_index: int,
        reference_frame_path: str = None,
        product_image_path: str = None
    ) -> str:
        """
        Generate a unique image for the segment.
        Uses role and style information to create appropriate visuals.
        If reference_frame_path is provided, uses img2img for style-consistent generation.
        """
        filename = f"{job_id}_segment_{segment_index}.png"
        filepath = os.path.join(self.output_dir, filename)
        
        # Enhance the prompt based on role and style
        enhanced_prompt = self._enhance_prompt_for_role(
            visual_prompt, role, visual_style, emotion
        )
        
        # Try AI generation if enabled
        if self.use_ai_generation:
            try:
                # Use dual-reference img2img if we have both trending frame and product image
                if reference_frame_path and os.path.exists(reference_frame_path):
                    if product_image_path and os.path.exists(product_image_path):
                        print(f"    → Using dual-reference img2img (trending style + product image)")
                        ai_image = self._generate_dual_reference_img2img(
                            enhanced_prompt,
                            reference_frame_path,  # Style reference (trending video)
                            product_image_path,  # Product reference
                            strength=self.img2img_strength,
                            role=role  # Pass role for better compositing
                        )
                    else:
                        # Single reference: trending video frame only
                        print(f"    → Using img2img with reference frame: {os.path.basename(reference_frame_path)}")
                        ai_image = self._generate_with_huggingface_img2img(
                            enhanced_prompt, 
                            reference_frame_path,
                            strength=self.img2img_strength
                        )
                else:
                    # Fall back to text2img
                    print(f"    → Using text2img (no reference frame)")
                    ai_image = self._generate_with_huggingface(enhanced_prompt)
                
                if ai_image:
                    ai_image = self._resize_and_crop_to_canvas(ai_image)
                    ai_image.save(filepath)
                    gen_type = "img2img" if reference_frame_path else "text2img"
                    print(f"    ✓ AI image generated ({gen_type}) for segment {segment_index + 1}")
                    self.generated_images.append(filepath)
                    return filepath
            except Exception as e:
                print(f"    ⚠ AI generation failed: {e}, using placeholder")
        
        # Fallback to enhanced placeholder
        return self._generate_enhanced_placeholder(
            visual_prompt=visual_prompt,
            role=role,
            visual_style=visual_style,
            emotion=emotion,
            filepath=filepath,
            segment_index=segment_index
        )
    
    def _enhance_prompt_for_role(
        self,
        visual_prompt: str,
        role: str,
        visual_style: str,
        emotion: str
    ) -> str:
        """
        Enhance the visual prompt based on segment role and style.
        Now includes technical photography terms for better AI generation.
        """
        # Role-specific prompt enhancements with technical terms
        role_styles = {
            "HOOK": "eye-catching, attention-grabbing, bold, dynamic composition, dramatic lighting, high contrast, shallow depth of field, close-up or medium shot, vibrant colors, cinematic",
            "PROBLEM": "relatable, slightly chaotic, realistic scenario, natural lighting, medium depth of field, medium shot, warm color temperature, documentary-style, authentic",
            "BUILDUP": "anticipation, building tension, transitional, soft lighting, medium depth of field, rule of thirds composition, cool color tones, cinematic",
            "SOLUTION": "clean, satisfying, product hero shot, spotlight effect, soft front lighting or dramatic side lighting, shallow depth of field, close-up, high contrast, commercial photography style, aspirational",
            "RESULT": "aspirational, success, happy outcome, bright, soft natural lighting, wide shot or medium shot, warm color temperature, lifestyle photography, positive mood",
            "CTA": "urgent, bold text, clear action, vibrant colors, high contrast, eye-level camera angle, commercial style, clear product visibility"
        }
        
        # Emotion-specific enhancements with technical terms
        emotion_styles = {
            "curiosity": "mysterious, intriguing lighting, low-key lighting, shadows and highlights, dramatic contrast, cinematic atmosphere",
            "frustration": "warm tones (3000K color temperature), cluttered composition, realistic lighting, medium contrast, relatable setting",
            "excitement": "vibrant colors, high saturation, dynamic composition, bright lighting, high contrast, energetic atmosphere",
            "satisfaction": "warm glow (soft 3000K lighting), organized composition, soft lighting, low contrast, peaceful atmosphere",
            "relief": "soft lighting, calm atmosphere, low contrast, cool color tones, peaceful composition",
            "urgency": "bold colors, high contrast, attention-grabbing, dramatic lighting, high saturation, commercial style"
        }
        
        # Camera angle and shot type based on role
        role_camera = {
            "HOOK": "eye-level or low-angle camera, close-up or medium shot",
            "PROBLEM": "eye-level camera, medium shot, natural perspective",
            "BUILDUP": "eye-level or slight high-angle, medium shot, transitional framing",
            "SOLUTION": "eye-level or slight low-angle, close-up or medium shot, product-focused",
            "RESULT": "eye-level camera, wide shot or medium shot, lifestyle framing",
            "CTA": "eye-level camera, medium shot, clear product visibility"
        }
        
        role_style = role_styles.get(role, "professional product photography, commercial style, eye-level camera, medium shot")
        emotion_style = emotion_styles.get(emotion.lower(), "neutral lighting, balanced composition")
        camera_info = role_camera.get(role, "eye-level camera, medium shot")
        
        # Build enhanced prompt with technical terms
        enhanced = f"""Professional product advertisement photography, {visual_prompt}. Style: {visual_style}. Mood: {role_style}, {emotion_style}.
Camera: {camera_info}.
High quality, detailed, commercial photography style, 9:16 vertical format (1080x1920), social media ready, trending aesthetic, sharp focus, professional lighting, cinematic composition, rule of thirds, balanced exposure"""
        
        # Add style consistency hints if we have overall style
        if self.current_style:
            color_palette = self.current_style.get("color_palette", {})
            if color_palette and isinstance(color_palette, dict):
                desc = color_palette.get("description", "")
                primary = color_palette.get("primary_colors", [])
                if desc:
                    enhanced += f", color scheme: {desc}"
                if primary:
                    dominant = primary[0] if primary else None
                    if dominant:
                        enhanced += f", dominant colors: {dominant}"
            elif isinstance(color_palette, str):
                enhanced += f", color palette: {color_palette}"
        
        return enhanced
    
    def _generate_with_huggingface(self, prompt: str) -> Optional[Image.Image]:
        """
        Generate image using Hugging Face Inference API with Stable Diffusion XL.
        Text-to-Image generation.
        """
        headers = {
            "Authorization": f"Bearer {self.hf_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "negative_prompt": self.negative_prompt,
                "num_inference_steps": self.sdxl_steps,
                "guidance_scale": self.sdxl_guidance,
                "width": self.sdxl_width,
                "height": self.sdxl_height
            }
        }
        if self.sdxl_seed:
            try:
                payload["parameters"]["seed"] = int(self.sdxl_seed)
            except ValueError:
                pass
        
        print(f"    → Generating (text2img): {prompt[:60]}...")
        
        response = requests.post(
            self.hf_model_url,
            headers=headers,
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            return image
        elif response.status_code == 503:
            print("    → Model loading, waiting 20s...")
            import time
            time.sleep(20)
            response = requests.post(
                self.hf_model_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                return image
            else:
                print(f"    → Retry failed: {response.status_code}")
                return None
        else:
            print(f"    → API error: {response.status_code}")
            return None
    
    def _generate_with_huggingface_img2img(
        self, 
        prompt: str, 
        reference_image_path: str,
        strength: float = 0.6
    ) -> Optional[Image.Image]:
        """
        Generate image using Hugging Face Inference API with img2img.
        Uses a reference image as the base and transforms it based on the prompt.
        
        Args:
            prompt: Text prompt describing what to generate
            reference_image_path: Path to the reference image from the trending video
            strength: How much to deviate from reference (0.0 = keep original, 1.0 = ignore original)
                      Recommended: 0.4-0.7 for good balance of style transfer
        
        Returns:
            Generated PIL Image or None if failed
        """
        import base64
        import time
        
        # Load and prepare reference image
        try:
            ref_image = Image.open(reference_image_path)
            
            # Resize reference image to max 1024px for API efficiency
            max_size = 1024
            if max(ref_image.size) > max_size:
                ratio = max_size / max(ref_image.size)
                new_size = (int(ref_image.size[0] * ratio), int(ref_image.size[1] * ratio))
                ref_image = ref_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary (handle RGBA images)
            if ref_image.mode != 'RGB':
                ref_image = ref_image.convert('RGB')
            
            # Encode image as base64
            buffer = io.BytesIO()
            ref_image.save(buffer, format='JPEG', quality=90)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
        except Exception as e:
            print(f"    ⚠ Failed to load reference image: {e}")
            # Fall back to text2img
            return self._generate_with_huggingface(prompt)
        
        headers = {
            "Authorization": f"Bearer {self.hf_api_key}",
            "Content-Type": "application/json"
        }
        
        # Enhance prompt to guide the transformation
        img2img_prompt = f"""Transform this image to show: {prompt}
Keep the same composition and layout but replace the content with the product.
Maintain similar colors, lighting, and style."""
        
        # Try using the base SDXL model with image input
        # Hugging Face img2img format
        payload = {
            "inputs": img2img_prompt,
            "parameters": {
                "negative_prompt": self.negative_prompt,
                "num_inference_steps": max(35, self.sdxl_steps),
                "guidance_scale": self.sdxl_guidance,
                "strength": strength,  # Key parameter for img2img
                "image": image_base64  # Reference image as base64
            }
        }
        if self.sdxl_seed:
            try:
                payload["parameters"]["seed"] = int(self.sdxl_seed)
            except ValueError:
                pass
        
        print(f"    → Generating (img2img, strength={strength}): {prompt[:50]}...")
        
        # Try the main model URL first (some models support img2img)
        response = requests.post(
            self.hf_model_url,
            headers=headers,
            json=payload,
            timeout=150  # Longer timeout for img2img
        )
        
        if response.status_code == 200:
            try:
                image = Image.open(io.BytesIO(response.content))
                return image
            except Exception as e:
                print(f"    ⚠ Failed to parse img2img response: {e}")
        
        elif response.status_code == 503:
            print("    → Model loading, waiting 25s...")
            time.sleep(25)
            response = requests.post(
                self.hf_model_url,
                headers=headers,
                json=payload,
                timeout=150
            )
            if response.status_code == 200:
                try:
                    image = Image.open(io.BytesIO(response.content))
                    return image
                except Exception as e:
                    print(f"    ⚠ Failed to parse img2img response: {e}")
        
        elif response.status_code in [400, 422]:
            # API may not support img2img in this format, try alternative approach
            print(f"    → img2img format not supported, trying alternative...")
            return self._generate_img2img_alternative(prompt, ref_image, strength)
        
        else:
            print(f"    → img2img API error: {response.status_code}")
            # Try alternative method
            return self._generate_img2img_alternative(prompt, ref_image, strength)
        
        # If all else fails, fall back to text2img
        print("    → Falling back to text2img...")
        return self._generate_with_huggingface(prompt)
    
    def _generate_img2img_alternative(
        self,
        prompt: str,
        reference_image: Image.Image,
        strength: float = 0.6
    ) -> Optional[Image.Image]:
        """
        Alternative img2img approach: Generate with text2img, then blend with reference.
        This is a fallback when the API doesn't support native img2img.
        
        Creates a style-consistent image by:
        1. Generating a new image with text2img
        2. Blending it with the reference image based on strength
        """
        print("    → Using blend-based img2img alternative...")
        
        # Generate new image with text2img
        new_image = self._generate_with_huggingface(prompt)
        
        if new_image is None:
            return None
        
        try:
            # Resize reference to match generated image
            ref_resized = reference_image.resize(new_image.size, Image.Resampling.LANCZOS)
            
            # Convert both to RGB if needed
            if new_image.mode != 'RGB':
                new_image = new_image.convert('RGB')
            if ref_resized.mode != 'RGB':
                ref_resized = ref_resized.convert('RGB')
            
            # Blend the images based on strength
            # strength = 1.0 means 100% new image, 0.0 means 100% reference
            # We want higher strength to show more of the product (new image)
            blended = Image.blend(ref_resized, new_image, strength)
            
            # Apply slight sharpening to improve quality
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Sharpness(blended)
            blended = enhancer.enhance(1.2)
            
            # Enhance contrast slightly
            contrast = ImageEnhance.Contrast(blended)
            blended = contrast.enhance(1.1)
            
            return blended
            
        except Exception as e:
            print(f"    ⚠ Blend failed: {e}, returning text2img result")
            return new_image
    
    def _generate_dual_reference_img2img(
        self,
        prompt: str,
        style_reference_path: str,  # Trending video frame
        product_reference_path: str,  # Product image
        strength: float = 0.6,
        role: str = "CONTENT"  # Add role for better compositing
    ) -> Optional[Image.Image]:
        """
        Generate image using both style reference (trending video) and product reference.
        Strategy: Generate with style reference, then composite/blend product into it.
        """
        from PIL import Image, ImageEnhance, ImageFilter
        
        try:
            # Load references
            style_ref = Image.open(style_reference_path).convert('RGB')
            product_ref = Image.open(product_reference_path).convert('RGB')
            
            # Step 1: Generate base image with style reference (trending video style)
            base_image = self._generate_with_huggingface_img2img(
                prompt,
                style_reference_path,
                strength=0.7  # Higher strength to allow more product adaptation
            )
            
            if base_image is None:
                return None
            
            # Step 2: Resize product to fit scene (smart sizing based on role)
            product_resized = self._smart_resize_product(product_ref, base_image.size, role)
            
            # Step 3: Better compositing with role-based blend ratio
            # Role-based blend ratios: SOLUTION/RESULT need more product visibility
            blend_ratios = {
                "HOOK": 0.25,      # Less product, more style
                "PROBLEM": 0.2,    # Minimal product
                "BUILDUP": 0.25,   # Building up
                "SOLUTION": 0.4,   # More product visibility
                "RESULT": 0.35,    # Good product visibility
                "CTA": 0.3         # Balanced
            }
            blend_ratio = blend_ratios.get(role, 0.3)
            
            # Use alpha compositing for better quality
            # Create product with transparency mask
            product_with_alpha = self._prepare_product_with_alpha(product_resized, base_image.size)
            
            # Composite product onto base with better blending
            if product_with_alpha.mode == 'RGBA':
                # Use alpha compositing for better quality
                base_rgba = base_image.convert('RGBA')
                # Adjust alpha based on blend ratio
                alpha_adjusted = self._adjust_alpha(product_with_alpha, blend_ratio)
                blended = Image.alpha_composite(base_rgba, alpha_adjusted).convert('RGB')
            else:
                # Fallback to simple blend
                blended = Image.blend(base_image, product_resized, blend_ratio)
            
            # Step 4: Enhance to maintain quality
            enhancer = ImageEnhance.Sharpness(blended)
            blended = enhancer.enhance(1.15)  # Slightly more sharpening
            
            # Add subtle color matching
            blended = self._match_colors(blended, style_ref, strength=0.2)
            
            return blended
            
        except Exception as e:
            print(f"    ⚠ Dual-reference failed: {e}, falling back to style-only")
            return self._generate_with_huggingface_img2img(prompt, style_reference_path, strength)
    
    def _smart_resize_product(self, product_img: Image.Image, target_size: tuple, role: str = "CONTENT") -> Image.Image:
        """Resize product image intelligently to fit target size while maintaining aspect ratio."""
        target_w, target_h = target_size
        prod_w, prod_h = product_img.size
        
        # Role-based sizing: SOLUTION/RESULT can be larger, HOOK smaller
        size_ratios = {
            "HOOK": 0.25,      # Smaller for hook
            "PROBLEM": 0.2,    # Small
            "BUILDUP": 0.25,   # Small-medium
            "SOLUTION": 0.4,   # Larger for product focus
            "RESULT": 0.35,    # Medium-large
            "CTA": 0.3         # Medium
        }
        size_ratio = size_ratios.get(role, 0.3)
        
        # Resize product based on role
        scale = min(target_w * size_ratio / prod_w, target_h * size_ratio / prod_h)
        new_size = (int(prod_w * scale), int(prod_h * scale))
        
        return product_img.resize(new_size, Image.Resampling.LANCZOS)
    
    def _prepare_product_with_alpha(self, product_img: Image.Image, target_size: tuple) -> Image.Image:
        """Prepare product image with alpha channel for better compositing."""
        # Create RGBA version
        if product_img.mode != 'RGBA':
            product_rgba = product_img.convert('RGBA')
        else:
            product_rgba = product_img.copy()
        
        # Create a new image with target size, centered product
        result = Image.new('RGBA', target_size, (0, 0, 0, 0))
        
        # Center the product
        x = (target_size[0] - product_rgba.width) // 2
        y = (target_size[1] - product_rgba.height) // 2
        
        result.paste(product_rgba, (x, y), product_rgba)
        
        return result
    
    def _adjust_alpha(self, img: Image.Image, blend_ratio: float) -> Image.Image:
        """Adjust alpha channel based on blend ratio."""
        if img.mode != 'RGBA':
            return img
        
        # Convert to array for manipulation
        img_array = np.array(img)
        
        # Adjust alpha channel
        img_array[:, :, 3] = (img_array[:, :, 3] * blend_ratio).astype(np.uint8)
        
        return Image.fromarray(img_array, 'RGBA')
    
    def _match_colors(self, target_img: Image.Image, reference_img: Image.Image, strength: float = 0.2) -> Image.Image:
        """Match colors from reference image to target (subtle color grading)."""
        try:
            from PIL import ImageStat
            
            # Get average colors
            ref_stat = ImageStat.Stat(reference_img.resize((100, 100)))
            target_stat = ImageStat.Stat(target_img.resize((100, 100)))
            
            # Calculate color shift
            ref_mean = np.array(ref_stat.mean)
            target_mean = np.array(target_stat.mean)
            
            # Apply subtle color shift
            shift = (ref_mean - target_mean) * strength
            shifted = target_img.copy()
            
            # Apply color shift using ImageEnhance
            # This is a simplified approach - full color matching would use LAB color space
            if shift[0] > 0:  # Red shift
                enhancer = ImageEnhance.Color(shifted)
                shifted = enhancer.enhance(1.0 + shift[0] / 255.0 * 0.1)
            
            return shifted
            
        except Exception as e:
            # If color matching fails, return original
            return target_img
    
    def _select_product_image_for_segment(self, role: str, segment_index: int) -> Optional[str]:
        """
        Smart product image selection: Match product images to segment roles.
        Strategy: Use filename hints, image analysis, or role-based selection.
        """
        if not self.product_image_paths:
            return None
        
        if len(self.product_image_paths) == 1:
            return self.product_image_paths[0]
        
        # Try to match by filename hints (e.g., "product_closeup.jpg", "product_lifestyle.jpg")
        role_keywords = {
            "HOOK": ["hook", "hero", "main", "featured", "highlight"],
            "PROBLEM": ["problem", "before", "issue", "pain"],
            "SOLUTION": ["solution", "product", "closeup", "detail", "feature"],
            "RESULT": ["result", "after", "lifestyle", "use", "happy", "success"],
            "CTA": ["cta", "logo", "brand", "final"]
        }
        
        keywords = role_keywords.get(role, [])
        
        # Check filenames for role keywords
        for img_path in self.product_image_paths:
            filename_lower = os.path.basename(img_path).lower()
            for keyword in keywords:
                if keyword in filename_lower:
                    return img_path
        
        # Fallback: Role-based selection strategy
        # HOOK/SOLUTION: Use first image (usually hero shot)
        # PROBLEM: Use middle images
        # RESULT: Use last images (usually lifestyle/result shots)
        # CTA: Use first image (logo/brand)
        
        num_images = len(self.product_image_paths)
        
        if role == "HOOK" or role == "SOLUTION" or role == "CTA":
            return self.product_image_paths[0]
        elif role == "PROBLEM":
            # Use middle image(s)
            return self.product_image_paths[num_images // 2]
        elif role == "RESULT":
            # Use last image(s) for lifestyle/result shots
            return self.product_image_paths[-1]
        else:
            # Default: cycle through
            return self.product_image_paths[segment_index % num_images]
    
    def _resize_and_crop_to_canvas(self, image: Image.Image) -> Image.Image:
        """
        Resize and crop image to fit 9:16 canvas (1080x1920).
        """
        target_width, target_height = self.canvas_size
        target_ratio = target_width / target_height
        
        img_width, img_height = image.size
        img_ratio = img_width / img_height
        
        if img_ratio > target_ratio:
            new_height = img_height
            new_width = int(img_height * target_ratio)
            left = (img_width - new_width) // 2
            top = 0
            right = left + new_width
            bottom = img_height
        else:
            new_width = img_width
            new_height = int(img_width / target_ratio)
            left = 0
            top = (img_height - new_height) // 2
            right = img_width
            bottom = top + new_height
        
        image = image.crop((left, top, right, bottom))
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        return image
    
    def _generate_enhanced_placeholder(
        self,
        visual_prompt: str,
        role: str,
        visual_style: str,
        emotion: str,
        filepath: str,
        segment_index: int
    ) -> str:
        """
        Generate enhanced placeholder image with role-specific styling.
        """
        # Role-based color schemes
        role_colors = {
            "HOOK": ((45, 45, 80), (255, 200, 100)),      # Dark blue bg, gold accent
            "PROBLEM": ((60, 40, 40), (200, 150, 150)),   # Dark red bg, muted accent
            "BUILDUP": ((40, 50, 60), (150, 180, 200)),   # Dark teal bg, light accent
            "SOLUTION": ((30, 60, 50), (100, 255, 180)),  # Dark green bg, bright accent
            "RESULT": ((50, 50, 70), (200, 200, 255)),    # Purple bg, light accent
            "CTA": ((80, 30, 30), (255, 100, 100))        # Red bg, bright accent
        }
        
        bg_color, accent_color = role_colors.get(role, ((40, 40, 50), (200, 200, 200)))
        
        # Create gradient background
        img = self._create_gradient_background(bg_color, segment_index)
        draw = ImageDraw.Draw(img)
        
        # Add decorative elements based on role
        self._add_role_decoration(draw, role, accent_color)
        
        # Add role badge
        self._add_role_badge(draw, role, accent_color)
        
        # Add visual prompt text
        self._add_prompt_text(draw, visual_prompt, accent_color)
        
        # Add segment number
        self._add_segment_number(draw, segment_index, accent_color)
        
        img.save(filepath)
        return filepath
    
    def _create_gradient_background(
        self,
        base_color: tuple,
        seed: int
    ) -> Image.Image:
        """Create a gradient background"""
        img = Image.new('RGB', self.canvas_size, base_color)
        draw = ImageDraw.Draw(img)
        
        # Add gradient effect
        for y in range(self.canvas_size[1]):
            progress = y / self.canvas_size[1]
            r = int(base_color[0] * (1 - progress * 0.3))
            g = int(base_color[1] * (1 - progress * 0.3))
            b = int(base_color[2] * (1 - progress * 0.2))
            draw.line([(0, y), (self.canvas_size[0], y)], fill=(r, g, b))
        
        return img
    
    def _add_role_decoration(
        self,
        draw: ImageDraw.Draw,
        role: str,
        accent_color: tuple
    ):
        """Add role-specific decorative elements"""
        w, h = self.canvas_size
        
        if role == "HOOK":
            # Burst/starburst pattern
            for i in range(12):
                angle = i * 30
                import math
                x1 = w // 2 + int(200 * math.cos(math.radians(angle)))
                y1 = h // 3 + int(200 * math.sin(math.radians(angle)))
                x2 = w // 2 + int(350 * math.cos(math.radians(angle)))
                y2 = h // 3 + int(350 * math.sin(math.radians(angle)))
                draw.line([(x1, y1), (x2, y2)], fill=accent_color, width=3)
        
        elif role == "PROBLEM":
            # Chaotic lines
            for i in range(5):
                y = 400 + i * 150
                draw.line([(100, y), (w - 100, y + 50)], fill=accent_color, width=2)
        
        elif role == "SOLUTION":
            # Spotlight effect
            for i in range(10):
                offset = i * 40
                alpha = int(255 * (1 - i/10))
                draw.ellipse(
                    [w//2 - 200 - offset, h//2 - 200 - offset,
                     w//2 + 200 + offset, h//2 + 200 + offset],
                    outline=(*accent_color[:3],),
                    width=2
                )
        
        elif role == "CTA":
            # Arrow pointing down
            arrow_y = h - 400
            draw.polygon([
                (w//2, arrow_y + 100),
                (w//2 - 80, arrow_y),
                (w//2 + 80, arrow_y)
            ], fill=accent_color)
        
        # Add border
        draw.rectangle([20, 20, w-20, h-20], outline=accent_color, width=4)
    
    def _add_role_badge(
        self,
        draw: ImageDraw.Draw,
        role: str,
        accent_color: tuple
    ):
        """Add role badge at the top"""
        font = self._get_font(32)
        
        badge_text = f"[ {role} ]"
        bbox = draw.textbbox((0, 0), badge_text, font=font)
        text_width = bbox[2] - bbox[0]
        
        x = (self.canvas_size[0] - text_width) // 2
        y = 60
        
        # Badge background
        padding = 20
        draw.rectangle(
            [x - padding, y - 10, x + text_width + padding, y + 40],
            fill=accent_color,
            outline=(255, 255, 255),
            width=2
        )
        
        # Badge text
        draw.text((x, y), badge_text, font=font, fill=(0, 0, 0))
    
    def _add_prompt_text(
        self,
        draw: ImageDraw.Draw,
        visual_prompt: str,
        accent_color: tuple
    ):
        """Add wrapped visual prompt text"""
        font = self._get_font(36)
        small_font = self._get_font(28)
        
        # Title
        title = "VISUAL PROMPT"
        bbox = draw.textbbox((0, 0), title, font=small_font)
        title_width = bbox[2] - bbox[0]
        draw.text(
            ((self.canvas_size[0] - title_width) // 2, 600),
            title,
            font=small_font,
            fill=accent_color
        )
        
        # Wrap text
        max_width = self.canvas_size[0] - 160
        words = visual_prompt.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        
        # Draw wrapped text
        line_height = 46
        start_y = 680
        
        for i, line in enumerate(lines[:8]):  # Limit to 8 lines
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x = (self.canvas_size[0] - text_width) // 2
            y = start_y + i * line_height
            
            # Shadow
            draw.text((x + 2, y + 2), line, font=font, fill=(0, 0, 0))
            # Main text
            draw.text((x, y), line, font=font, fill=(255, 255, 255))
    
    def _add_segment_number(
        self,
        draw: ImageDraw.Draw,
        segment_index: int,
        accent_color: tuple
    ):
        """Add segment number at the bottom"""
        font = self._get_font(48)
        
        text = f"SEGMENT {segment_index + 1}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        
        x = (self.canvas_size[0] - text_width) // 2
        y = self.canvas_size[1] - 200
        
        draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=accent_color)
    
    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Get a font, with fallback to default"""
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            "arial.ttf",
            "Arial.ttf",
        ]
        
        for font_path in font_paths:
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
        
        return ImageFont.load_default()
    
    def _animate_segment(
        self,
        image_path: str,
        motion: str,
        duration: float,
        overlay_text: Optional[str],
        role: str,
        emotion: str
    ) -> CompositeVideoClip:
        """
        Create animated video clip with role-appropriate motion effects.
        """
        # Optional depth-based parallax animation for more realistic motion.
        img_clip = None
        if self.enable_depth_parallax:
            try:
                img_clip = self._create_parallax_clip(image_path, duration)
            except Exception as e:
                print(f"    ⚠ Parallax failed: {e}, falling back to 2D motion")
                img_clip = None

        if img_clip is None:
            img_clip = ImageClip(image_path, duration=duration)
        
        if img_clip.w != self.canvas_size[0] or img_clip.h != self.canvas_size[1]:
            img_clip = img_clip.resize(self.canvas_size)
        
        # Background canvas
        bg_clip = ColorClip(size=self.canvas_size, color=(20, 20, 30), duration=duration)
        composite_clips = [bg_clip]
        
        # Apply role-specific motion
        img_clip = self._apply_role_motion(img_clip, motion, role, duration)
        composite_clips.append(img_clip)
        
        # Add overlay text if present
        if overlay_text:
            txt_clip = self._create_text_overlay(overlay_text, duration, role)
            if txt_clip:
                composite_clips.append(txt_clip)
        
        return CompositeVideoClip(composite_clips, size=self.canvas_size)

    def _init_midas(self):
        """Lazy-load MiDaS for depth estimation."""
        if self._midas is not None:
            return
        try:
            import torch
            device = os.environ.get("MIDAS_DEVICE")
            if not device:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._midas_device = device

            self._midas = torch.hub.load("intel-isl/MiDaS", self.midas_model)
            self._midas.eval()
            self._midas.to(self._midas_device)

            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if "small" in self.midas_model.lower():
                self._midas_transform = transforms.small_transform
            else:
                self._midas_transform = transforms.default_transform
        except Exception as e:
            raise RuntimeError(f"MiDaS init failed: {e}")

    def _estimate_depth(self, image_rgb: np.ndarray) -> np.ndarray:
        """Estimate a normalized depth map (0..1) for an RGB image."""
        import torch
        self._init_midas()
        input_batch = self._midas_transform(image_rgb).to(self._midas_device)
        with torch.no_grad():
            prediction = self._midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_rgb.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()
        depth = prediction.cpu().numpy()
        # Normalize to 0..1
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min < 1e-6:
            return np.zeros_like(depth, dtype=np.float32)
        depth_norm = (depth - depth_min) / (depth_max - depth_min)
        return depth_norm.astype(np.float32)

    def _create_parallax_clip(self, image_path: str, duration: float) -> VideoClip:
        """Create a depth-based parallax clip from a single image."""
        import cv2
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        if img.size != self.canvas_size:
            img = img.resize(self.canvas_size, Image.Resampling.LANCZOS)
        img_np = np.array(img)
        depth = self._estimate_depth(img_np)

        h, w = depth.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid_x = grid_x.astype(np.float32)
        grid_y = grid_y.astype(np.float32)

        amp_x = self.parallax_amplitude
        amp_y = self.parallax_vertical

        def make_frame(t):
            # Oscillate parallax over time for subtle motion
            phase = np.sin(2 * np.pi * (t / max(duration, 0.1)))
            shift_x = (depth - 0.5) * amp_x * phase
            shift_y = (depth - 0.5) * amp_y * phase
            map_x = grid_x + shift_x
            map_y = grid_y + shift_y
            warped = cv2.remap(
                img_np,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
            return warped

        return VideoClip(make_frame=make_frame, duration=duration)

    def _apply_parallax_quality_defaults(self) -> None:
        """
        Apply quality-tier defaults for parallax to balance speed vs quality.
        This only sets defaults if specific env overrides are not provided.
        """
        if not self.parallax_quality:
            return

        # Only override if user did not specify explicit values
        midas_override = "MIDAS_MODEL" not in os.environ
        amp_override = "PARALLAX_AMPLITUDE" not in os.environ
        vert_override = "PARALLAX_VERTICAL" not in os.environ

        if self.parallax_quality in ("low", "fast"):
            if midas_override:
                self.midas_model = "midas_v21_small_256"
            if amp_override:
                self.parallax_amplitude = 10.0
            if vert_override:
                self.parallax_vertical = 5.0
        elif self.parallax_quality in ("medium", "balanced"):
            if midas_override:
                self.midas_model = "midas_v21_small_256"
            if amp_override:
                self.parallax_amplitude = 14.0
            if vert_override:
                self.parallax_vertical = 7.0
        elif self.parallax_quality in ("high", "quality"):
            if midas_override:
                self.midas_model = "dpt_hybrid_384"
            if amp_override:
                self.parallax_amplitude = 16.0
            if vert_override:
                self.parallax_vertical = 8.0
    
    def _apply_role_motion(
        self,
        clip: ImageClip,
        motion: str,
        role: str,
        duration: float
    ) -> ImageClip:
        """Apply motion effects based on role and motion type"""
        if self.motion_overscan > 1.0:
            clip = clip.resize(self.motion_overscan)
        
        # Role-specific motion adjustments
        if role == "HOOK":
            # Attention-grabbing: quick zoom in
            return clip.resize(lambda t: 1.0 + 0.15 * min(t / 0.5, 1)).set_position('center')
        
        elif role == "PROBLEM":
            # Subtle shake: position is clip top-left; center at (0,0) + small offset
            def shake(t):
                offset_x = int(8 * np.sin(t * 12))
                offset_y = int(5 * np.cos(t * 10))
                return (offset_x, offset_y)
            return clip.set_position(shake)
        
        elif role == "SOLUTION":
            # Smooth zoom to product
            return clip.resize(lambda t: 1.0 + 0.2 * (t / duration)).set_position('center')
        
        elif role == "RESULT":
            # Gentle pan: position is clip top-left; use (0,0) + small pan so full image visible
            return clip.set_position(
                lambda t: (int(-30 * t / duration), 'center')
            )
        
        elif role == "CTA":
            # Pulse effect
            def pulse(t):
                scale = 1.0 + 0.05 * np.sin(t * 4 * np.pi)
                return scale
            return clip.resize(pulse).set_position('center')
        
        # Default motions
        elif motion == "static":
            return clip.set_position('center')
        elif motion == "slow_pan":
            # Pan: clip top-left from (0,0) so full image visible
            return clip.set_position(
                lambda t: (int(-50 * t / duration), 'center')
            )
        elif motion == "fast_zoom":
            return clip.resize(lambda t: 1.0 + 0.3 * t / duration).set_position('center')
        elif motion == "handheld":
            def shake(t):
                offset_x = int(10 * np.sin(t * 10))
                offset_y = int(10 * np.cos(t * 8))
                return (offset_x, offset_y)
            return clip.set_position(shake)
        else:
            return clip.set_position('center')
    
    def _create_text_overlay(
        self,
        text: str,
        duration: float,
        role: str
    ) -> Optional[TextClip]:
        """Create text overlay with role-appropriate styling, wrapped to stay inside 9:16."""
        try:
            # Role-specific text colors
            role_colors = {
                "HOOK": 'yellow',
                "PROBLEM": 'white',
                "SOLUTION": 'lime',
                "RESULT": 'white',
                "CTA": 'red'
            }
            
            color = role_colors.get(role, 'white')
            
            # Max text width to stay inside canvas with side margin (9:16 safe area)
            max_text_width = self.canvas_size[0] - 120
            
            # Position based on role with safe zones (keep text away from edges)
            # Safe zones: 60px from sides, 100px from top/bottom
            safe_top = 100
            safe_bottom = 100
            
            if role == "HOOK":
                # Top area for hooks
                position = ('center', safe_top + 50)
                fontsize = 56
            elif role == "CTA":
                # Bottom area for CTAs
                position = ('center', self.canvas_size[1] - safe_bottom - 60)
                fontsize = 64
            else:
                # Middle-bottom for other roles
                position = ('center', self.canvas_size[1] - safe_bottom - 120)
                fontsize = 52
            
            # Use method='caption' to wrap text within max width so it stays in frame
            txt_clip = TextClip(
                text,
                fontsize=fontsize,
                color=color,
                font='Arial-Bold',
                stroke_color='black',
                stroke_width=3,
                method='caption',
                size=(max_text_width, None)
            ).set_duration(duration)

            # Clamp Y position so text stays inside canvas bounds
            x_pos, y_pos = position
            if isinstance(y_pos, int):
                max_y = self.canvas_size[1] - safe_bottom - txt_clip.h
                y_pos = max(safe_top, min(y_pos, max_y))
            position = (x_pos, y_pos)
            txt_clip = txt_clip.set_position(position)
            
            # Add fade-in animation (fade in over first 0.3 seconds)
            fade_duration = min(0.3, duration * 0.2)  # 20% of duration or 0.3s max
            txt_clip = txt_clip.fadein(fade_duration)
            
            # Add subtle slide-in from bottom for non-HOOK segments
            if role != "HOOK":
                # Start slightly below, slide up
                def slide_position(t):
                    if t < fade_duration:
                        # Slide up during fade-in
                        progress = t / fade_duration
                        offset = int(20 * (1 - progress))  # Start 20px down, end at 0
                        x, y = position
                        if isinstance(y, int):
                            return (x, y + offset)
                        return position
                    return position
                txt_clip = txt_clip.set_position(slide_position)
            
            return txt_clip
            
        except Exception as e:
            print(f"    ⚠ Text overlay failed: {e}")
            return None
