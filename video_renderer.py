"""
Video Renderer - Enhanced with Transitions and Beat-Sync
Assembles the final video with smooth transitions, beat synchronization,
and styled text overlays matching the trend aesthetic.
"""
import os
from typing import Dict, List, Any, Optional
import uuid
from moviepy.editor import (
    VideoFileClip, ImageClip, CompositeVideoClip,
    TextClip, AudioFileClip, concatenate_videoclips,
    concatenate_audioclips, ColorClip, vfx
)
import numpy as np


class VideoRenderer:
    """
    Enhanced video renderer that creates polished final videos with:
    - Smooth transitions between segments
    - Beat-synchronized cuts
    - Styled text overlays
    - Professional CTA animations
    """
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.canvas_size = (1080, 1920)
    
    def render_video(
        self,
        ad_blueprint: Dict[str, Any],
        animated_video_paths: List[str],
        concept_blueprint: Dict[str, Any],
        job_id: str,
        original_audio_path: str,
        product_logo_path: Optional[str] = None
    ) -> str:
        """
        Render final video from animated segments with enhanced features.
        """
        print(f"  → Rendering final video with {len(animated_video_paths)} segments...")
        
        # Get sync points for beat-matched cuts
        audio_info = concept_blueprint.get("audio", {})
        sync_points = audio_info.get("sync_points", [])
        beat_times = audio_info.get("beat_times", [])
        
        # Load all segment clips
        clips = []
        for i, video_path in enumerate(animated_video_paths):
            if os.path.exists(video_path):
                clip = VideoFileClip(video_path)
                
                # Ensure correct size
                if clip.size != list(self.canvas_size):
                    clip = clip.resize(self.canvas_size)
                
                clips.append(clip)
            else:
                print(f"  ⚠ Missing segment video: {video_path}")
        
        if not clips:
            raise ValueError("No valid video clips to render")
        
        # Apply transitions between clips
        print("  → Applying transitions...")
        clips_with_transitions = self._apply_transitions(
            clips, 
            ad_blueprint,
            sync_points
        )
        
        # Concatenate clips
        final_clip = concatenate_videoclips(clips_with_transitions, method="compose")
        
        # Add original audio
        audio = None
        if os.path.exists(original_audio_path):
            print("  → Adding original audio...")
            audio = AudioFileClip(original_audio_path)
            
            # Normalize audio levels for consistent volume
            audio = self._normalize_audio(audio)
            
            # Match audio duration to video
            if audio.duration > final_clip.duration:
                audio = audio.subclip(0, final_clip.duration)
            elif audio.duration < final_clip.duration:
                # Loop audio if needed
                loops_needed = int(final_clip.duration / audio.duration) + 1
                audio_clips = [audio] * loops_needed
                audio = concatenate_audioclips(audio_clips).subclip(0, final_clip.duration)
            
            final_clip = final_clip.set_audio(audio)
        
        # Add CTA if present
        cta_text = ad_blueprint.get("cta")
        if cta_text:
            print("  → Adding CTA...")
            final_clip = self._add_animated_cta(final_clip, cta_text)
        
        # Add logo overlay if provided
        if product_logo_path and os.path.exists(product_logo_path):
            print("  → Adding product logo overlay...")
            final_clip = self._add_logo_overlay(final_clip, product_logo_path)
        
        # Render output
        output_filename = f"{job_id}_final.mp4"
        output_path = os.path.join(self.output_dir, output_filename)
        
        print("  → Encoding final video...")
        final_clip.write_videofile(
            output_path,
            fps=30,
            codec='libx264',
            audio_codec='aac',
            preset='slow',  # Better quality encoding
            bitrate='12000k',  # Higher bitrate for better quality
            ffmpeg_params=['-profile:v', 'high', '-level', '4.0'],  # High profile for better quality
            verbose=False,
            logger=None
        )
        
        # Clean up
        final_clip.close()
        for clip in clips:
            clip.close()
        if audio is not None:
            audio.close()
        
        return output_path
    
    def _apply_transitions(
        self,
        clips: List[VideoFileClip],
        ad_blueprint: Dict[str, Any],
        sync_points: List[float]
    ) -> List[VideoFileClip]:
        """
        Apply transitions between clips based on segment roles and style.
        """
        if len(clips) <= 1:
            return clips
        
        # Get overall style preferences
        overall_style = ad_blueprint.get("overall_style", {})
        transition_style = overall_style.get("transitions", "cut").lower()
        
        # Get segments info
        segments = ad_blueprint.get("segments", ad_blueprint.get("shots", []))
        
        result_clips = []
        
        for i, clip in enumerate(clips):
            # Determine transition type based on segment role
            if i < len(segments):
                current_role = segments[i].get("role", "CONTENT")
                next_role = segments[i + 1].get("role", "CONTENT") if i + 1 < len(segments) else None
            else:
                current_role = "CONTENT"
                next_role = None
            
            # Apply role-based transitions
            if i < len(clips) - 1:
                transition_type = self._get_transition_type(current_role, next_role, transition_style)
                clip = self._apply_transition_effect(clip, transition_type, "out")
            
            if i > 0:
                prev_role = segments[i - 1].get("role", "CONTENT") if i - 1 < len(segments) else "CONTENT"
                transition_type = self._get_transition_type(prev_role, current_role, transition_style)
                clip = self._apply_transition_effect(clip, transition_type, "in")
            
            result_clips.append(clip)
        
        return result_clips
    
    def _get_transition_type(
        self,
        from_role: str,
        to_role: str,
        default_style: str
    ) -> str:
        """Determine transition type based on segment roles"""
        
        # Role-based transition mapping
        transition_map = {
            ("HOOK", "PROBLEM"): "cut",           # Sharp cut for impact
            ("PROBLEM", "BUILDUP"): "slide",      # Slide transition
            ("BUILDUP", "SOLUTION"): "zoom",      # Dramatic zoom reveal
            ("SOLUTION", "RESULT"): "crossfade",  # Satisfying transition
            ("RESULT", "CTA"): "wipe",            # Wipe to CTA
            ("*", "CTA"): "flash",                # Always flash to CTA
        }
        
        # Check specific transitions
        key = (from_role, to_role)
        if key in transition_map:
            return transition_map[key]
        
        # Check wildcard transitions
        wildcard_key = ("*", to_role)
        if wildcard_key in transition_map:
            return transition_map[wildcard_key]
        
        # Default based on style
        if "fast" in default_style:
            return "cut"
        elif "smooth" in default_style:
            return "crossfade"
        else:
            return "cut"
    
    def _apply_transition_effect(
        self,
        clip: VideoFileClip,
        transition_type: str,
        direction: str  # "in" or "out"
    ) -> VideoFileClip:
        """Apply transition effect to clip start or end"""
        
        transition_duration = 0.3  # seconds
        
        if transition_type == "cut":
            # No effect needed for hard cut
            return clip
        
        elif transition_type == "crossfade":
            if direction == "in":
                return clip.crossfadein(transition_duration)
            else:
                return clip.crossfadeout(transition_duration)
        
        elif transition_type == "fade":
            if direction == "in":
                return clip.fadein(transition_duration)
            else:
                return clip.fadeout(transition_duration)
        
        elif transition_type == "zoom":
            # Zoom transition effect
            if direction == "out":
                # Zoom out at the end
                return clip.resize(lambda t: 1.0 + 0.3 * max(0, (t - (clip.duration - transition_duration)) / transition_duration))
            elif direction == "in":
                # Zoom in at the start
                return clip.resize(lambda t: 1.3 - 0.3 * min(1, t / transition_duration))
            return clip
        
        elif transition_type == "slide":
            # Slide transition (slide left/right)
            if direction == "out":
                # Slide out to the left
                return clip.set_position(lambda t: (
                    int(-self.canvas_size[0] * max(0, (t - (clip.duration - transition_duration)) / transition_duration)),
                    'center'
                ))
            elif direction == "in":
                # Slide in from the right
                return clip.set_position(lambda t: (
                    int(self.canvas_size[0] * (1 - min(1, t / transition_duration))),
                    'center'
                ))
            return clip
        
        elif transition_type == "wipe":
            # Wipe transition (horizontal wipe)
            if direction == "out":
                # Wipe out to the left (fade out with position shift)
                def wipe_out(t):
                    clip_duration = clip.duration
                    if t > clip_duration - transition_duration:
                        progress = (t - (clip_duration - transition_duration)) / transition_duration
                        # Combine fade with slight position shift
                        return (int(-self.canvas_size[0] * 0.1 * progress), 'center')
                    return ('center', 'center')
                return clip.set_position(wipe_out).fadeout(transition_duration)
            elif direction == "in":
                # Wade in from the right
                def wipe_in(t):
                    if t < transition_duration:
                        progress = t / transition_duration
                        return (int(self.canvas_size[0] * 0.1 * (1 - progress)), 'center')
                    return ('center', 'center')
                return clip.set_position(wipe_in).fadein(transition_duration)
            return clip
        
        elif transition_type == "flash":
            # White flash transition
            if direction == "in":
                # Create white flash at the start with simple fade out
                flash_duration = 0.15
                flash = ColorClip(
                    size=self.canvas_size,
                    color=(255, 255, 255),
                    duration=flash_duration
                )
                # Use fadeout instead of lambda opacity
                flash = flash.fadeout(flash_duration)
                
                return CompositeVideoClip([clip, flash.set_start(0)], size=self.canvas_size)
            return clip
        
        return clip
    
    def _add_animated_cta(
        self,
        video_clip: VideoFileClip,
        cta_text: str,
        cta_duration: float = 3.0
    ) -> CompositeVideoClip:
        """
        Add animated CTA overlay at the end of the video.
        """
        video_duration = video_clip.duration
        
        # CTA appears in the last few seconds
        cta_start = max(0, video_duration - cta_duration)
        
        try:
            # Create CTA text (simple, no animation to avoid opacity issues)
            max_text_width = self.canvas_size[0] - 120
            txt_clip = TextClip(
                cta_text,
                fontsize=64,
                color='white',
                font='Arial-Bold',
                stroke_color='black',
                stroke_width=3,
                method='caption',
                size=(max_text_width, None)
            )
            
            # Position at bottom center with padding, clamp to stay inside canvas
            y_pos = self.canvas_size[1] - 200
            max_y = self.canvas_size[1] - 100 - txt_clip.h
            y_pos = max(100, min(y_pos, max_y))
            txt_clip = txt_clip.set_position(('center', y_pos))
            txt_clip = txt_clip.set_start(cta_start).set_duration(cta_duration)
            
            # Simple fade in instead of complex opacity animation
            txt_clip = txt_clip.crossfadein(0.3)
            
            # Create CTA background bar with static opacity
            bar_height = 120
            bar_clip = ColorClip(
                size=(self.canvas_size[0], bar_height),
                color=(0, 0, 0),
                duration=cta_duration
            )
            # Use static opacity value (not a function)
            bar_clip = bar_clip.set_opacity(0.7)
            bar_clip = bar_clip.set_position(('center', self.canvas_size[1] - 250))
            bar_clip = bar_clip.set_start(cta_start)
            
            # Fade in the bar
            bar_clip = bar_clip.crossfadein(0.3)
            
            # Composite
            return CompositeVideoClip(
                [video_clip, bar_clip, txt_clip],
                size=self.canvas_size
            )
            
        except Exception as e:
            print(f"  ⚠ CTA creation failed: {e}")
            return video_clip
    
    def _add_logo_overlay(
        self,
        video_clip: VideoFileClip,
        logo_path: str
    ) -> CompositeVideoClip:
        """Add product logo overlay at top-right corner."""
        from PIL import Image
        
        try:
            # Load and prepare logo
            logo_img = Image.open(logo_path)
            
            # Resize logo to reasonable size (e.g., 15% of video width)
            logo_width = int(self.canvas_size[0] * 0.15)  # 15% of width
            logo_height = int(logo_width * (logo_img.height / logo_img.width))
            logo_resized = logo_img.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
            
            # Save resized logo temporarily
            temp_logo_path = f"temp/{uuid.uuid4()}_logo.png"
            os.makedirs("temp", exist_ok=True)
            logo_resized.save(temp_logo_path, "PNG")
            
            # Create logo clip
            logo_clip = ImageClip(temp_logo_path, duration=video_clip.duration)
            
            # Set opacity to 30% as requested
            logo_clip = logo_clip.set_opacity(0.30)
            
            # Add fade-in animation (fade in over first 1 second)
            fade_duration = min(1.0, video_clip.duration * 0.1)
            logo_clip = logo_clip.fadein(fade_duration)
            
            # Position at top-right with padding
            # Logo is positioned absolutely on top of video - doesn't affect video position
            padding = 30
            logo_position = (
                self.canvas_size[0] - logo_width - padding,
                padding
            )
            logo_clip = logo_clip.set_position(logo_position)
            
            # Video clip is already sized to canvas_size, so it fills the entire canvas
            # Logo is positioned absolutely on top as an overlay
            # In CompositeVideoClip, order determines z-index: later clips = higher z-index
            # [video_clip, logo_clip] means logo appears on top (higher z-index)
            # Video position remains unchanged - logo is just an overlay layer
            return CompositeVideoClip([video_clip, logo_clip], size=self.canvas_size)
            
        except Exception as e:
            print(f"  ⚠ Logo overlay failed: {e}")
            return video_clip
    
    def _normalize_audio(self, audio_clip: AudioFileClip) -> AudioFileClip:
        """
        Normalize audio levels to ensure consistent volume.
        Uses peak normalization to target -3dB peak level.
        """
        try:
            # Get audio array
            audio_array = audio_clip.to_soundarray(fps=audio_clip.fps)
            
            if audio_array.size == 0:
                return audio_clip
            
            # Find peak level
            max_level = np.max(np.abs(audio_array))
            
            if max_level > 0:
                # Target peak level: -3dB (0.707 of max)
                target_peak = 0.707
                # Calculate gain needed
                gain = target_peak / max_level
                # Limit gain to prevent over-amplification (max 3x)
                gain = min(gain, 3.0)
                
                if gain != 1.0:
                    # Apply gain
                    audio_array = audio_array * gain
                    # Create new audio clip from normalized array
                    from moviepy.audio.AudioClip import AudioArrayClip
                    normalized_audio = AudioArrayClip(audio_array, fps=audio_clip.fps)
                    normalized_audio = normalized_audio.set_duration(audio_clip.duration)
                    return normalized_audio
            
            return audio_clip
            
        except Exception as e:
            print(f"  ⚠ Audio normalization failed: {e}, using original audio")
            return audio_clip
    
    def _create_beat_synced_cuts(
        self,
        clips: List[VideoFileClip],
        beat_times: List[float],
        total_duration: float
    ) -> List[VideoFileClip]:
        """
        Adjust clip timings to sync cuts with beat times.
        This is an advanced feature for precise beat-matching.
        """
        if not beat_times or len(clips) <= 1:
            return clips
        
        # Calculate current clip boundaries
        boundaries = []
        current_time = 0
        for clip in clips:
            boundaries.append(current_time)
            current_time += clip.duration
        
        # Find nearest beat for each boundary and adjust
        adjusted_clips = []
        for i, clip in enumerate(clips):
            if i == 0:
                adjusted_clips.append(clip)
                continue
            
            # Find nearest beat to this clip's start
            clip_start = boundaries[i]
            nearest_beat = min(beat_times, key=lambda b: abs(b - clip_start))
            
            # Calculate adjustment needed
            adjustment = nearest_beat - clip_start
            
            # Only adjust if within reasonable range (±0.2 seconds)
            if abs(adjustment) < 0.2:
                # Adjust previous clip duration
                prev_clip = adjusted_clips[-1]
                new_duration = prev_clip.duration + adjustment
                if new_duration > 0.5:  # Ensure minimum duration
                    adjusted_clips[-1] = prev_clip.set_duration(new_duration)
            
            adjusted_clips.append(clip)
        
        return adjusted_clips
    
    def render_preview(
        self,
        ad_blueprint: Dict[str, Any],
        animated_video_paths: List[str],
        job_id: str,
        preview_duration: float = 5.0
    ) -> str:
        """
        Render a quick preview of the first few seconds.
        Useful for testing without full rendering.
        """
        clips = []
        total_duration = 0
        
        for video_path in animated_video_paths:
            if os.path.exists(video_path) and total_duration < preview_duration:
                clip = VideoFileClip(video_path)
                remaining = preview_duration - total_duration
                
                if clip.duration > remaining:
                    clip = clip.subclip(0, remaining)
                
                clips.append(clip)
                total_duration += clip.duration
                
                if total_duration >= preview_duration:
                    break
        
        if not clips:
            raise ValueError("No clips for preview")
        
        preview_clip = concatenate_videoclips(clips, method="compose")
        
        output_path = os.path.join(self.output_dir, f"{job_id}_preview.mp4")
        preview_clip.write_videofile(
            output_path,
            fps=30,
            codec='libx264',
            preset='ultrafast',
            verbose=False,
            logger=None
        )
        
        preview_clip.close()
        for clip in clips:
            clip.close()
        
        return output_path


if __name__ == "__main__":
    # Test renderer
    renderer = VideoRenderer()
