"""
Style Extractor - Enhanced Video Analysis
Combines Vision AI analysis with audio analysis for comprehensive understanding.
Now uses GPT-4 Vision to understand the "concept" of trending videos.
"""
import cv2
import librosa
import numpy as np
from typing import Dict, List, Any, Optional
import json
import os
from moviepy.editor import VideoFileClip

# Import new analyzers
from vision_analyzer import VisionAnalyzer
from smart_segmenter import SmartSegmenter
from audio_analyzer import AudioAnalyzer


class StyleExtractor:
    """
    Enhanced style extractor that uses Vision AI to understand video concepts.
    Falls back to basic extraction if Vision AI is not available.
    """
    
    def __init__(self, use_vision_ai: bool = True, use_whisper: bool = True):
        """
        Initialize the style extractor.
        
        Args:
            use_vision_ai: Whether to use GPT-4 Vision for analysis (default True)
            use_whisper: Whether to use Whisper for speech transcription (default True)
        """
        self.use_vision_ai = use_vision_ai
        self.use_whisper = use_whisper
        self.motion_threshold = 0.1
        
        # Initialize Vision Analyzer if enabled
        self.vision_analyzer = None
        if use_vision_ai:
            try:
                self.vision_analyzer = VisionAnalyzer()
                print("✓ Vision AI analyzer initialized")
            except Exception as e:
                print(f"⚠ Vision AI not available: {e}")
                self.use_vision_ai = False
        
        # Initialize Audio Analyzer with Whisper
        self.audio_analyzer = None
        if use_whisper:
            try:
                self.audio_analyzer = AudioAnalyzer(use_whisper_api=True)
                print("✓ Audio Analyzer with Whisper initialized")
            except Exception as e:
                print(f"⚠ Audio Analyzer not available: {e}")
                self.use_whisper = False
        
        # Initialize Smart Segmenter
        self.segmenter = SmartSegmenter(min_segments=3, max_segments=10)
    
    def extract_style(self, video_path: str, job_id: str = None) -> Dict[str, Any]:
        """
        Main extraction function.
        Returns enhanced Concept Blueprint if Vision AI is available,
        otherwise falls back to basic Style Blueprint.
        
        Args:
            video_path: Path to the video file
            job_id: Optional job ID for unique keyframe naming (for img2img)
        """
        # Get video duration first
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        # Extract audio features - use enhanced analyzer if available
        print("  → Extracting audio features...")
        if self.audio_analyzer:
            audio_data = self.audio_analyzer.analyze_audio(video_path)
        else:
            audio_data = self._extract_audio_features(video_path)
        
        if self.use_vision_ai and self.vision_analyzer:
            # Use enhanced Vision AI analysis
            return self._extract_with_vision_ai(video_path, audio_data, total_duration, job_id)
        else:
            # Fallback to basic extraction
            return self._extract_basic(video_path, audio_data, total_duration)
    
    def _extract_with_vision_ai(
        self,
        video_path: str,
        audio_data: Dict[str, Any],
        total_duration: float,
        job_id: str = None
    ) -> Dict[str, Any]:
        """
        Enhanced extraction using Vision AI.
        Returns a comprehensive Concept Blueprint with saved keyframe paths for img2img.
        """
        print("  → Analyzing video with GPT-4 Vision...")
        
        # Get vision analysis (with keyframe saving for img2img)
        vision_analysis = self.vision_analyzer.analyze_video(video_path, max_frames=8, job_id=job_id)
        
        # Create smart segments
        print("  → Creating semantic segments...")
        segments = self.segmenter.create_segments(
            vision_analysis,
            audio_data,
            total_duration
        )
        
        # Build the concept blueprint
        concept_blueprint = self.segmenter.create_concept_blueprint(
            vision_analysis,
            audio_data,
            segments,
            total_duration
        )
        
        # Add legacy fields for backward compatibility
        concept_blueprint["hook_duration"] = segments[0]["duration"] if segments else 1.0
        concept_blueprint["shots"] = self._convert_segments_to_shots(segments)
        concept_blueprint["music"] = {
            "bpm": audio_data.get("bpm", 120),
            "beat_drop": audio_data.get("beat_drop", 1.0)
        }
        
        return concept_blueprint
    
    def _convert_segments_to_shots(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert new segment format to legacy shot format for backward compatibility.
        """
        shots = []
        for segment in segments:
            shot = {
                "start": segment["start"],
                "end": segment["end"],
                "duration": segment["duration"],
                "motion": self._emotion_to_motion(segment.get("emotion", "neutral")),
                "text_overlay": segment.get("text_content") is not None,
                "energy": self._role_to_energy(segment.get("role", "HOOK")),
                # New fields
                "role": segment.get("role"),
                "scene_content": segment.get("scene_content"),
                "visual_style": segment.get("visual_style"),
                "text_content": segment.get("text_content"),
                "emotion": segment.get("emotion"),
                "has_beat_sync": segment.get("has_beat_sync", False)
            }
            shots.append(shot)
        return shots
    
    def _emotion_to_motion(self, emotion: str) -> str:
        """Map emotion to motion type"""
        emotion = emotion.lower()
        if "excite" in emotion or "energy" in emotion or "fast" in emotion:
            return "fast_zoom"
        elif "calm" in emotion or "peace" in emotion or "slow" in emotion:
            return "slow_pan"
        elif "dynamic" in emotion or "action" in emotion:
            return "handheld"
        else:
            return "static"
    
    def _role_to_energy(self, role: str) -> str:
        """Map narrative role to energy level"""
        high_energy = ["HOOK", "CTA", "SOLUTION"]
        low_energy = ["PROBLEM", "BUILDUP"]
        
        if role in high_energy:
            return "high"
        elif role in low_energy:
            return "low"
        else:
            return "medium"
    
    def _extract_basic(
        self,
        video_path: str,
        audio_data: Dict[str, Any],
        total_duration: float
    ) -> Dict[str, Any]:
        """
        Basic extraction without Vision AI (fallback).
        Returns traditional Style Blueprint.
        """
        print("  → Using basic visual analysis (no Vision AI)...")
        
        # Extract visual features
        visual_data = self._extract_visual_features(video_path)
        
        # Combine into style blueprint
        style_blueprint = {
            "total_duration": float(total_duration),
            "hook_duration": float(visual_data.get("hook_duration", 1.0)),
            "shots": visual_data["shots"],
            "music": audio_data,
            # Add placeholder trend analysis for compatibility
            "trend_analysis": {
                "why_trending": "Unknown (Vision AI not used)",
                "hook_technique": "Unknown",
                "narrative_formula": "Unknown",
                "visual_style": "Unknown",
                "key_elements": [],
                "content_type": "unknown",
                "pacing": "medium",
                "text_usage": "unknown"
            },
            "segments": self._shots_to_segments(visual_data["shots"])
        }
        
        return style_blueprint
    
    def _shots_to_segments(self, shots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert basic shots to segment format"""
        segments = []
        for i, shot in enumerate(shots):
            segment = {
                "id": i + 1,
                "start": shot["start"],
                "end": shot["end"],
                "duration": shot["duration"],
                "scene_content": f"Scene {i + 1}",
                "visual_style": "Standard",
                "text_content": None,
                "emotion": "neutral",
                "has_beat_sync": False,
                "role": self._assign_basic_role(i, len(shots))
            }
            segments.append(segment)
        return segments
    
    def _assign_basic_role(self, index: int, total: int) -> str:
        """Assign basic narrative role based on position"""
        if index == 0:
            return "HOOK"
        elif index == total - 1:
            return "CTA"
        elif index < total / 2:
            return "PROBLEM"
        else:
            return "SOLUTION"
    
    def _extract_visual_features(self, video_path: str) -> Dict[str, Any]:
        """Extract visual style features (basic method)"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps if fps > 0 else 0
        
        # Detect shot boundaries
        shots = self._detect_shot_boundaries(cap, fps, total_duration)
        
        # Analyze each shot
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        analyzed_shots = []
        
        for i, shot in enumerate(shots):
            shot_data = self._analyze_shot(cap, shot, fps)
            analyzed_shots.append(shot_data)
        
        cap.release()
        
        return {
            "total_duration": float(total_duration),
            "hook_duration": float(shots[0]["duration"]) if shots else 1.0,
            "shots": analyzed_shots
        }
    
    def _detect_shot_boundaries(
        self,
        cap: cv2.VideoCapture,
        fps: float,
        total_duration: float
    ) -> List[Dict]:
        """
        Detect shot boundaries using frame difference analysis.
        Enhanced to ensure minimum segments.
        """
        shots = []
        prev_frame = None
        shot_start_frame = 0
        frame_count = 0
        
        # Lower threshold to detect more subtle changes
        cut_threshold = 0.25
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                diff = cv2.absdiff(gray, prev_frame)
                diff_score = np.mean(diff) / 255.0
                
                if diff_score > cut_threshold:
                    shot_end_frame = frame_count - 1
                    shot_duration = (shot_end_frame - shot_start_frame) / fps
                    
                    if shot_duration > 0.1:
                        shots.append({
                            "start": float(shot_start_frame / fps),
                            "end": float(shot_end_frame / fps),
                            "duration": float(shot_duration)
                        })
                    
                    shot_start_frame = frame_count
            
            prev_frame = gray
            frame_count += 1
        
        # Add final shot
        if shot_start_frame < frame_count:
            shot_duration = (frame_count - shot_start_frame) / fps
            if shot_duration > 0.1:
                shots.append({
                    "start": float(shot_start_frame / fps),
                    "end": float(frame_count / fps),
                    "duration": float(shot_duration)
                })
        
        # IMPORTANT: If we only detected 1-2 shots, force more segments
        if len(shots) < 3 and total_duration > 3:
            print("  → Too few shots detected, creating artificial segments...")
            shots = self._create_artificial_segments(total_duration)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return shots
    
    def _create_artificial_segments(self, total_duration: float) -> List[Dict]:
        """
        Create artificial segments when shot detection fails.
        Creates 4-6 segments based on duration.
        """
        num_segments = min(6, max(4, int(total_duration / 2.5)))
        segment_duration = total_duration / num_segments
        
        shots = []
        for i in range(num_segments):
            start = i * segment_duration
            end = (i + 1) * segment_duration
            shots.append({
                "start": float(start),
                "end": float(end),
                "duration": float(segment_duration)
            })
        
        return shots
    
    def _analyze_shot(self, cap: cv2.VideoCapture, shot: Dict, fps: float) -> Dict:
        """Analyze a single shot for motion, text overlay, and framing."""
        start_frame = int(shot["start"] * fps)
        end_frame = int(shot["end"] * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        
        for frame_num in range(start_frame, min(end_frame, start_frame + 30)):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        if not frames:
            return {
                "start": float(shot["start"]),
                "end": float(shot["end"]),
                "duration": float(shot["duration"]),
                "motion": "static",
                "text_overlay": False,
                "energy": "medium"
            }
        
        motion = self._detect_motion_type(frames)
        text_overlay = self._detect_text_overlay(frames[0])
        energy = self._estimate_energy(motion, shot["duration"])
        
        return {
            "start": float(shot["start"]),
            "end": float(shot["end"]),
            "duration": float(shot["duration"]),
            "motion": motion,
            "text_overlay": text_overlay,
            "energy": energy
        }
    
    def _detect_motion_type(self, frames: List[np.ndarray]) -> str:
        """Detect camera motion type"""
        if len(frames) < 2:
            return "static"
        
        motion_scores = []
        for i in range(1, len(frames)):
            gray1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray1, gray2)
            motion_score = np.mean(diff) / 255.0
            motion_scores.append(motion_score)
        
        avg_motion = np.mean(motion_scores)
        
        if avg_motion < 0.05:
            return "static"
        elif avg_motion < 0.15:
            return "slow_pan"
        elif avg_motion < 0.3:
            return "handheld"
        else:
            return "fast_zoom"
    
    def _detect_text_overlay(self, frame: np.ndarray) -> bool:
        """Detect if frame has text overlay"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        white_ratio = np.sum(thresh > 0) / (gray.shape[0] * gray.shape[1])
        return bool(white_ratio > 0.05)
    
    def _estimate_energy(self, motion: str, duration: float) -> str:
        """Estimate energy level"""
        fast_motions = ["fast_zoom", "handheld"]
        slow_motions = ["static", "slow_pan"]
        
        if motion in fast_motions and duration < 1.5:
            return "high"
        elif motion in slow_motions and duration > 2.0:
            return "low"
        else:
            return "medium"
    
    def _extract_audio_features(self, video_path: str) -> Dict[str, Any]:
        """Extract audio features: BPM, beat drops, etc."""
        try:
            y, sr = librosa.load(video_path, sr=None)
            duration = len(y) / sr
            
            # Calculate BPM
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            bpm = float(tempo)
            
            # Find beat times
            beat_times = librosa.frames_to_time(beats, sr=sr).tolist()
            beat_drop = float(beat_times[0]) if len(beat_times) > 0 else 1.0
            
            return {
                "bpm": round(bpm, 1),
                "beat_drop": round(beat_drop, 2),
                "beat_times": [round(t, 2) for t in beat_times],
                "duration": duration,
                "has_music": bpm > 60,
                "has_speech": False,  # Will be enhanced in Phase 2
                "speech_segments": [],
                "transcript": None
            }
        except Exception as e:
            print(f"Audio extraction failed: {e}")
            return {
                "bpm": 120.0,
                "beat_drop": 1.0,
                "beat_times": [],
                "duration": 0,
                "has_music": True,
                "has_speech": False,
                "speech_segments": [],
                "transcript": None
            }
    
    def extract_audio_file(self, video_path: str, output_path: str) -> str:
        """
        Extract and save original audio track from video.
        Returns path to saved audio file.
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            video = VideoFileClip(video_path)
            if video.audio is not None:
                audio = video.audio
                audio.write_audiofile(
                    output_path,
                    verbose=False,
                    logger=None
                )
                audio.close()
                video.close()
                return output_path
            else:
                video.close()
                raise ValueError("No audio track found in video")
        except Exception as e:
            print(f"Audio extraction failed: {e}")
            raise
    
    def save_blueprint(self, blueprint: Dict[str, Any], output_path: str):
        """Save blueprint to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(blueprint, f, indent=2)


if __name__ == "__main__":
    # Test extraction
    extractor = StyleExtractor(use_vision_ai=True)
    # blueprint = extractor.extract_style("test_video.mp4")
    # print(json.dumps(blueprint, indent=2))
