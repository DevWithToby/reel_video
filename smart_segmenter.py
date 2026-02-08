"""
Smart Segmenter - Semantic Video Segmentation
Creates meaningful segments from video analysis, combining:
- Vision analysis (scene changes, content shifts)
- Audio analysis (beats, speech, energy)
- Minimum segment rules (ensure enough variety)
"""
import json
from typing import Dict, List, Any, Optional
import numpy as np


class SmartSegmenter:
    """Creates semantic segments from video analysis for trend replication"""
    
    # Narrative roles that segments can have
    NARRATIVE_ROLES = ["HOOK", "PROBLEM", "BUILDUP", "SOLUTION", "RESULT", "CTA"]
    
    @staticmethod
    def _parse_timestamp(value) -> float:
        """
        Parse timestamp value which could be:
        - A float (e.g., 2.5)
        - An int (e.g., 2)
        - A string with 's' suffix (e.g., "2.5s" or "2.5 s")
        - A string number (e.g., "2.5")
        """
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Remove 's' suffix and whitespace
            cleaned = value.strip().rstrip('s').strip()
            try:
                return float(cleaned)
            except ValueError:
                return 0.0
        return 0.0
    
    def __init__(self, min_segments: int = 3, max_segments: int = 10):
        """
        Initialize the segmenter.
        
        Args:
            min_segments: Minimum number of segments to create (default 3)
            max_segments: Maximum number of segments (default 10)
        """
        self.min_segments = min_segments
        self.max_segments = max_segments
    
    def create_segments(
        self,
        vision_analysis: Dict[str, Any],
        audio_analysis: Dict[str, Any],
        total_duration: float
    ) -> List[Dict[str, Any]]:
        """
        Create semantic segments from video analysis.
        
        Args:
            vision_analysis: Output from VisionAnalyzer
            audio_analysis: Audio features (BPM, beats, speech segments)
            total_duration: Total video duration in seconds
        
        Returns:
            List of segment dictionaries with timing, role, and description
        """
        # Collect all potential segment boundaries
        boundaries = self._collect_boundaries(
            vision_analysis, 
            audio_analysis, 
            total_duration
        )
        
        # Merge nearby boundaries and enforce min/max segments
        boundaries = self._optimize_boundaries(boundaries, total_duration)
        
        # Create segments from boundaries
        segments = self._create_segment_objects(
            boundaries, 
            vision_analysis, 
            audio_analysis,
            total_duration
        )
        
        # Assign narrative roles
        segments = self._assign_narrative_roles(segments, vision_analysis)
        
        return segments
    
    def _collect_boundaries(
        self,
        vision_analysis: Dict[str, Any],
        audio_analysis: Dict[str, Any],
        total_duration: float
    ) -> List[Dict[str, Any]]:
        """
        Collect all potential segment boundaries from various sources.
        """
        boundaries = []
        
        # 1. Always start at 0
        boundaries.append({
            "time": 0.0,
            "source": "start",
            "confidence": 1.0
        })
        
        # 2. Frame timestamps from vision analysis (scene changes)
        frames = vision_analysis.get("frames", [])
        for frame in frames:
            ts = self._parse_timestamp(frame.get("timestamp", 0))
            if ts > 0:  # Skip the first frame (already at 0)
                boundaries.append({
                    "time": ts,
                    "source": "vision_frame",
                    "confidence": 0.7,
                    "frame_data": frame
                })
        
        # 3. Beat drops from audio (high-energy moments)
        beat_times = audio_analysis.get("beat_times", [])
        beat_drop = audio_analysis.get("beat_drop", 0)
        
        if beat_drop > 0:
            boundaries.append({
                "time": beat_drop,
                "source": "beat_drop",
                "confidence": 0.9
            })
        
        # Add significant beats (e.g., every 4th beat in first half)
        if beat_times:
            for i, beat_time in enumerate(beat_times):
                if i % 4 == 0 and beat_time > 0.5:  # Every 4th beat, after first 0.5s
                    boundaries.append({
                        "time": beat_time,
                        "source": "beat",
                        "confidence": 0.5
                    })
        
        # 4. Speech segment boundaries (if available)
        speech_segments = audio_analysis.get("speech_segments", [])
        for seg in speech_segments:
            boundaries.append({
                "time": seg.get("start", 0),
                "source": "speech_start",
                "confidence": 0.8,
                "text": seg.get("text", "")
            })
        
        # 5. Text appearance changes from vision
        for frame in frames:
            text_content = frame.get("text_content")
            if text_content and text_content != "null":
                boundaries.append({
                    "time": self._parse_timestamp(frame.get("timestamp", 0)),
                    "source": "text_change",
                    "confidence": 0.75,
                    "text": text_content
                })
        
        # Sort by time
        boundaries.sort(key=lambda x: x["time"])
        
        return boundaries
    
    def _optimize_boundaries(
        self,
        boundaries: List[Dict[str, Any]],
        total_duration: float
    ) -> List[float]:
        """
        Optimize boundaries: merge nearby ones, ensure min/max count.
        Returns list of final boundary timestamps.
        """
        if not boundaries:
            # Fallback: evenly distribute segments
            return self._create_even_boundaries(total_duration)
        
        # Merge boundaries that are too close (within 0.5 seconds)
        merge_threshold = 0.5
        merged = []
        
        for boundary in boundaries:
            time = boundary["time"]
            confidence = boundary.get("confidence", 0.5)
            
            # Check if this should merge with the last one
            if merged and abs(time - merged[-1]["time"]) < merge_threshold:
                # Keep the one with higher confidence
                if confidence > merged[-1].get("confidence", 0):
                    merged[-1] = boundary
            else:
                merged.append(boundary)
        
        # Extract just the times
        times = [b["time"] for b in merged]
        
        # Ensure we have at least min_segments
        if len(times) < self.min_segments:
            times = self._create_even_boundaries(total_duration)
        
        # Limit to max_segments (keep most confident)
        if len(times) > self.max_segments:
            # Sort by confidence and keep top ones
            scored = [(merged[i], i) for i in range(len(merged)) if i < len(merged)]
            scored.sort(key=lambda x: x[0].get("confidence", 0), reverse=True)
            kept_indices = sorted([s[1] for s in scored[:self.max_segments]])
            times = [merged[i]["time"] for i in kept_indices]
        
        # Always ensure 0 is included
        if times[0] != 0:
            times.insert(0, 0)
        
        return times
    
    def _create_even_boundaries(self, total_duration: float) -> List[float]:
        """Create evenly-spaced boundaries when no other data available"""
        num_segments = max(self.min_segments, min(5, int(total_duration / 2)))
        segment_duration = total_duration / num_segments
        return [i * segment_duration for i in range(num_segments)]
    
    def _create_segment_objects(
        self,
        boundaries: List[float],
        vision_analysis: Dict[str, Any],
        audio_analysis: Dict[str, Any],
        total_duration: float
    ) -> List[Dict[str, Any]]:
        """
        Create segment objects from boundaries.
        Includes reference_frame_path for img2img generation.
        """
        segments = []
        frames = vision_analysis.get("frames", [])
        
        for i, start_time in enumerate(boundaries):
            # Determine end time
            if i + 1 < len(boundaries):
                end_time = boundaries[i + 1]
            else:
                end_time = total_duration
            
            duration = end_time - start_time
            
            # Skip very short segments
            if duration < 0.3:
                continue
            
            # Find the closest frame for this segment
            closest_frame = self._find_closest_frame(frames, start_time)
            
            segment = {
                "id": i + 1,
                "start": round(start_time, 2),
                "end": round(end_time, 2),
                "duration": round(duration, 2),
                "scene_content": closest_frame.get("scene_content", "Scene") if closest_frame else "Scene",
                "visual_style": closest_frame.get("visual_style", "Standard") if closest_frame else "Standard",
                "text_content": closest_frame.get("text_content") if closest_frame else None,
                "emotion": closest_frame.get("emotion", "neutral") if closest_frame else "neutral",
                "has_beat_sync": self._check_beat_sync(start_time, audio_analysis),
                "role": None,  # Will be assigned later
                # Include reference frame path for img2img generation
                "reference_frame_path": closest_frame.get("frame_path") if closest_frame else None
            }
            
            segments.append(segment)
        
        return segments
    
    def _find_closest_frame(
        self,
        frames: List[Dict[str, Any]],
        timestamp: float
    ) -> Optional[Dict[str, Any]]:
        """Find the frame closest to the given timestamp"""
        if not frames:
            return None
        
        closest = None
        min_diff = float('inf')
        
        for frame in frames:
            frame_ts = self._parse_timestamp(frame.get("timestamp", 0))
            diff = abs(frame_ts - timestamp)
            if diff < min_diff:
                min_diff = diff
                closest = frame
        
        return closest
    
    def _check_beat_sync(
        self,
        timestamp: float,
        audio_analysis: Dict[str, Any]
    ) -> bool:
        """Check if timestamp aligns with a beat"""
        beat_times = audio_analysis.get("beat_times", [])
        beat_drop = audio_analysis.get("beat_drop", 0)
        
        # Check if within 0.1s of a beat
        for beat in beat_times:
            if abs(beat - timestamp) < 0.1:
                return True
        
        if abs(beat_drop - timestamp) < 0.1:
            return True
        
        return False
    
    def _assign_narrative_roles(
        self,
        segments: List[Dict[str, Any]],
        vision_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Assign narrative roles to segments based on position and content.
        """
        if not segments:
            return segments
        
        overall = vision_analysis.get("overall_analysis", {})
        narrative_formula = overall.get("narrative_formula", "")
        content_type = overall.get("content_type", "unknown")
        
        num_segments = len(segments)
        
        # Determine role assignment based on number of segments and content type
        if num_segments == 1:
            segments[0]["role"] = "HOOK"
        elif num_segments == 2:
            segments[0]["role"] = "HOOK"
            segments[1]["role"] = "CTA"
        elif num_segments == 3:
            segments[0]["role"] = "HOOK"
            segments[1]["role"] = "SOLUTION"
            segments[2]["role"] = "CTA"
        elif num_segments == 4:
            segments[0]["role"] = "HOOK"
            segments[1]["role"] = "PROBLEM"
            segments[2]["role"] = "SOLUTION"
            segments[3]["role"] = "CTA"
        elif num_segments == 5:
            segments[0]["role"] = "HOOK"
            segments[1]["role"] = "PROBLEM"
            segments[2]["role"] = "SOLUTION"
            segments[3]["role"] = "RESULT"
            segments[4]["role"] = "CTA"
        else:
            # For 6+ segments, distribute roles
            segments[0]["role"] = "HOOK"
            segments[-1]["role"] = "CTA"
            
            # Distribute middle roles
            middle_count = num_segments - 2
            middle_roles = ["PROBLEM", "BUILDUP", "SOLUTION", "RESULT"]
            
            for i in range(1, num_segments - 1):
                role_idx = min(i - 1, len(middle_roles) - 1)
                segments[i]["role"] = middle_roles[role_idx % len(middle_roles)]
        
        # Refine roles based on emotion/content
        for segment in segments:
            emotion = segment.get("emotion", "").lower()
            
            # Adjust role based on emotion hints
            if "frustrat" in emotion or "problem" in emotion or "struggle" in emotion:
                if segment["role"] not in ["HOOK", "CTA"]:
                    segment["role"] = "PROBLEM"
            elif "relief" in emotion or "satisf" in emotion or "happy" in emotion:
                if segment["role"] not in ["HOOK", "CTA"]:
                    segment["role"] = "RESULT"
            elif "curios" in emotion or "intrigue" in emotion:
                segment["role"] = "HOOK"
        
        return segments
    
    def create_concept_blueprint(
        self,
        vision_analysis: Dict[str, Any],
        audio_analysis: Dict[str, Any],
        segments: List[Dict[str, Any]],
        total_duration: float
    ) -> Dict[str, Any]:
        """
        Create the complete Concept Blueprint from all analyses.
        This is the enhanced version of the old "Style Blueprint".
        """
        overall = vision_analysis.get("overall_analysis", {})
        
        concept_blueprint = {
            "total_duration": total_duration,
            "trend_analysis": {
                "why_trending": overall.get("why_trending", "Unknown"),
                "hook_technique": overall.get("hook_technique", "Unknown"),
                "narrative_formula": overall.get("narrative_formula", "Hook → Content → CTA"),
                "visual_style": overall.get("visual_style_summary", "Standard"),
                "key_elements": overall.get("key_elements", []),
                "content_type": overall.get("content_type", "unknown"),
                "pacing": overall.get("pacing", "medium"),
                "text_usage": overall.get("text_usage", "moderate")
            },
            "segments": segments,
            "audio": {
                "bpm": audio_analysis.get("bpm"),
                "beat_drop": audio_analysis.get("beat_drop"),
                "has_speech": audio_analysis.get("has_speech", False),
                "has_music": audio_analysis.get("has_music", True),
                "transcript": audio_analysis.get("transcript"),
                "sync_points": self._extract_sync_points(audio_analysis, segments)
            },
            "frame_analysis": vision_analysis.get("frames", [])
        }
        
        return concept_blueprint
    
    def _extract_sync_points(
        self,
        audio_analysis: Dict[str, Any],
        segments: List[Dict[str, Any]]
    ) -> List[float]:
        """Extract key sync points for beat-matching"""
        sync_points = []
        
        # Add beat drop
        beat_drop = audio_analysis.get("beat_drop", 0)
        if beat_drop > 0:
            sync_points.append(beat_drop)
        
        # Add segment start times that align with beats
        for segment in segments:
            if segment.get("has_beat_sync"):
                sync_points.append(segment["start"])
        
        # Add first few beats
        beat_times = audio_analysis.get("beat_times", [])
        for beat in beat_times[:5]:  # First 5 beats
            if beat not in sync_points:
                sync_points.append(beat)
        
        return sorted(list(set(sync_points)))
    
    def save_blueprint(self, blueprint: Dict[str, Any], output_path: str):
        """Save concept blueprint to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(blueprint, f, indent=2)


if __name__ == "__main__":
    # Test the segmenter
    segmenter = SmartSegmenter()
    # Test with sample data
    pass
