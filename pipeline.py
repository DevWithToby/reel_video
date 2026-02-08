"""
Orchestration Pipeline - Enhanced with Vision AI and Smart Segmentation
Coordinates all steps: Vision Analysis → Audio Analysis → Smart Segmentation → 
LLM Creative → Multi-Asset Generation → Video Rendering
"""
import os
import json
from typing import Dict, Any, List, Optional
from style_extractor import StyleExtractor
from llm_director import LLMDirector
from asset_generator import AssetGenerator
from video_renderer import VideoRenderer
import random


class ReelPipeline:
    """
    Enhanced pipeline orchestrator that uses Vision AI to understand
    trending video concepts and replicate them for product ads.
    """
    
    def __init__(self, use_vision_ai: bool = True, use_whisper: bool = True):
        """
        Initialize the pipeline.
        
        Args:
            use_vision_ai: Whether to use GPT-4 Vision for video analysis
            use_whisper: Whether to use Whisper for speech transcription
        """
        self.style_extractor = StyleExtractor(
            use_vision_ai=use_vision_ai,
            use_whisper=use_whisper
        )
        self.llm_director = LLMDirector()
        self.asset_generator = AssetGenerator()
        self.video_renderer = VideoRenderer()
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
    
    def process(
        self,
        job_id: str,
        video_path: str,
        product_description: str,
        brand_tone: str = None,
        product_image_paths: List[str] = None,
        product_logo_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main processing pipeline.
        Returns dict with output video path and status.
        """
        try:
            # Step 1: Extract concept blueprint (Vision AI + Audio Analysis)
            print(f"[{job_id}] Step 1: Analyzing trending video concept...")
            print(f"  This includes:")
            print(f"    - GPT-4 Vision analysis of keyframes")
            print(f"    - Saving keyframes for img2img reference")
            print(f"    - Audio feature extraction (BPM, beats)")
            print(f"    - Whisper transcription (if speech detected)")
            print(f"    - Smart semantic segmentation")
            
            # Pass job_id to extract_style for unique keyframe naming
            concept_blueprint = self.style_extractor.extract_style(video_path, job_id=job_id)
            
            # Log what was detected
            self._log_concept_summary(job_id, concept_blueprint)
            
            # Step 1.5: Extract and save original audio
            print(f"[{job_id}] Step 1.5: Extracting original audio...")
            os.makedirs("temp", exist_ok=True)
            audio_output_path = os.path.join("temp", f"{job_id}_audio.wav")
            self.style_extractor.extract_audio_file(video_path, audio_output_path)
            
            # Step 2: Apply timing jitter (±10%) for uniqueness
            concept_blueprint = self._add_timing_jitter(concept_blueprint)
            
            # Log concept blueprint
            self._log_json(concept_blueprint, f"{job_id}_concept_blueprint.json")
            
            # Step 3: Generate ad blueprint with LLM (full concept context)
            print(f"[{job_id}] Step 2: Generating ad blueprint with GPT-4o...")
            print(f"  LLM now knows:")
            print(f"    - Why the video is trending")
            print(f"    - Hook technique to replicate")
            print(f"    - Narrative formula to follow")
            print(f"    - Visual style per segment")
            
            ad_blueprint = self.llm_director.generate_ad_blueprint(
                concept_blueprint,
                product_description,
                brand_tone
            )
            
            # Log ad blueprint
            self._log_json(ad_blueprint, f"{job_id}_ad_blueprint.json")
            
            # Log ad blueprint summary
            self._log_ad_summary(job_id, ad_blueprint)
            
            # Step 4: Generate animated assets (one per segment)
            num_segments = len(ad_blueprint.get("segments", ad_blueprint.get("shots", [])))
            segments_with_ref = sum(1 for s in ad_blueprint.get("segments", []) if s.get("reference_frame_path"))
            print(f"[{job_id}] Step 3: Generating {num_segments} unique segment visuals...")
            if segments_with_ref > 0:
                print(f"  → Using img2img with reference frames for {segments_with_ref}/{num_segments} segments")
            if product_image_paths:
                print(f"  → Using {len(product_image_paths)} product image(s) for accurate product representation")
            
            animated_video_paths = self.asset_generator.generate_assets(
                ad_blueprint, 
                job_id,
                product_image_paths=product_image_paths or []
            )
            
            # Step 5: Render final video with transitions and beat-sync
            print(f"[{job_id}] Step 4: Rendering video with transitions and beat-sync...")
            if product_logo_path:
                print(f"  → Adding product logo overlay")
            output_video_path = self.video_renderer.render_video(
                ad_blueprint,
                animated_video_paths,
                concept_blueprint,
                job_id,
                audio_output_path,
                product_logo_path=product_logo_path
            )
            
            print(f"[{job_id}] ✓ Complete! Output: {output_video_path}")
            
            return {
                "job_id": job_id,
                "status": "completed",
                "output_video_path": output_video_path,
                "concept_blueprint": concept_blueprint,
                "ad_blueprint": ad_blueprint,
                "segments_generated": num_segments
            }
        
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            print(f"[{job_id}] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "job_id": job_id,
                "status": "failed",
                "error": error_msg
            }
    
    def _log_concept_summary(self, job_id: str, concept_blueprint: Dict[str, Any]):
        """Log a summary of what was detected in the concept analysis"""
        trend = concept_blueprint.get("trend_analysis", {})
        segments = concept_blueprint.get("segments", [])
        audio = concept_blueprint.get("audio", {})
        
        print(f"\n{'='*60}")
        print(f"[{job_id}] CONCEPT ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"  Why Trending: {trend.get('why_trending', 'Unknown')[:80]}...")
        print(f"  Hook Technique: {trend.get('hook_technique', 'Unknown')[:60]}...")
        print(f"  Narrative Formula: {trend.get('narrative_formula', 'Unknown')}")
        print(f"  Content Type: {trend.get('content_type', 'Unknown')}")
        print(f"  Pacing: {trend.get('pacing', 'Unknown')}")
        print(f"  Text Usage: {trend.get('text_usage', 'Unknown')}")
        print(f"  Segments Detected: {len(segments)}")
        
        for seg in segments:
            print(f"    - {seg.get('role', '?')}: {seg.get('duration', 0):.1f}s - {seg.get('emotion', '?')}")
        
        print(f"  Audio Type: {audio.get('audio_type', 'Unknown')}")
        print(f"  BPM: {audio.get('bpm', 'Unknown')}")
        print(f"  Has Speech: {audio.get('has_speech', False)}")
        if audio.get('transcript'):
            print(f"  Transcript: {audio.get('transcript', '')[:100]}...")
        print(f"{'='*60}\n")
    
    def _log_ad_summary(self, job_id: str, ad_blueprint: Dict[str, Any]):
        """Log a summary of the generated ad blueprint"""
        segments = ad_blueprint.get("segments", ad_blueprint.get("shots", []))
        trend_rep = ad_blueprint.get("trend_replication", {})
        
        print(f"\n{'='*60}")
        print(f"[{job_id}] AD BLUEPRINT SUMMARY")
        print(f"{'='*60}")
        if trend_rep:
            print(f"  Original Hook: {trend_rep.get('original_hook', 'N/A')[:60]}...")
            print(f"  Adapted Hook: {trend_rep.get('adapted_hook', 'N/A')[:60]}...")
        
        print(f"  Segments Generated: {len(segments)}")
        for seg in segments:
            print(f"    - {seg.get('role', '?')}: {seg.get('duration', 0):.1f}s")
            print(f"      Visual: {seg.get('visual_prompt', 'N/A')[:50]}...")
            if seg.get('overlay_text'):
                print(f"      Text: {seg.get('overlay_text')[:40]}...")
        
        print(f"  CTA: {ad_blueprint.get('cta', 'N/A')}")
        print(f"{'='*60}\n")
    
    def _add_timing_jitter(self, concept_blueprint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add ±10% jitter to timing for uniqueness.
        Ensures we don't exactly copy timing patterns.
        """
        jitter_factor = random.uniform(0.9, 1.1)
        
        # Adjust total duration
        if "total_duration" in concept_blueprint:
            concept_blueprint["total_duration"] *= jitter_factor
        
        # Adjust hook duration
        if "hook_duration" in concept_blueprint:
            concept_blueprint["hook_duration"] *= jitter_factor
        
        # Adjust segments
        for segment in concept_blueprint.get("segments", []):
            seg_jitter = random.uniform(0.9, 1.1)
            segment["duration"] *= seg_jitter
            segment["start"] *= jitter_factor
            segment["end"] *= jitter_factor
        
        # Also adjust shots for backward compatibility
        for shot in concept_blueprint.get("shots", []):
            shot_jitter = random.uniform(0.9, 1.1)
            shot["duration"] *= shot_jitter
            if "start" in shot:
                shot["start"] *= jitter_factor
            if "end" in shot:
                shot["end"] *= jitter_factor
        
        return concept_blueprint
    
    def _log_json(self, data: Dict[str, Any], filename: str):
        """Log JSON data for debugging and compliance"""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Logged: {filepath}")


# For backward compatibility
def create_pipeline(use_vision_ai: bool = True, use_whisper: bool = True) -> ReelPipeline:
    """Factory function to create a pipeline with specified options"""
    return ReelPipeline(use_vision_ai=use_vision_ai, use_whisper=use_whisper)


if __name__ == "__main__":
    # Test pipeline
    pipeline = ReelPipeline()
    # result = pipeline.process("test_job", "test_video.mp4", "A cool product", "professional")
    # print(result)