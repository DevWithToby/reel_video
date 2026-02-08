"""
Audio Analyzer - Enhanced Audio Analysis
Uses Whisper for transcription and speech detection.
Analyzes music, beats, energy, and speech content.
"""
import os
import json
import librosa
import numpy as np
from typing import Dict, List, Any, Optional
from openai import OpenAI


class AudioAnalyzer:
    """
    Enhanced audio analyzer that uses Whisper for transcription
    and librosa for music/beat analysis.
    """
    
    def __init__(self, api_key: Optional[str] = None, use_whisper_api: bool = True):
        """
        Initialize the audio analyzer.
        
        Args:
            api_key: OpenAI API key (for Whisper API)
            use_whisper_api: Whether to use OpenAI Whisper API (default True)
        """
        self.use_whisper_api = use_whisper_api
        self.client = None
        self.whisper_model = os.getenv("WHISPER_MODEL", "whisper-1")
        self.whisper_sample_rate = int(os.getenv("WHISPER_SAMPLE_RATE", "48000"))
        self.whisper_format = os.getenv("WHISPER_AUDIO_FORMAT", "wav")
        
        if use_whisper_api:
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
                print("✓ Whisper API initialized for speech transcription")
            else:
                print("⚠ No OpenAI API key - speech transcription disabled")
                self.use_whisper_api = False
    
    def analyze_audio(self, video_path: str) -> Dict[str, Any]:
        """
        Complete audio analysis including music features and speech transcription.
        
        Args:
            video_path: Path to the video file
        
        Returns:
            Dictionary with all audio analysis results
        """
        print("  → Analyzing audio features...")
        
        # Extract basic audio features with librosa
        music_features = self._analyze_music_features(video_path)
        
        # Transcribe speech with Whisper
        speech_data = {"has_speech": False, "transcript": None, "speech_segments": []}
        if self.use_whisper_api and self.client:
            print("  → Transcribing speech with Whisper...")
            speech_data = self._transcribe_with_whisper(video_path)
        
        # Combine results
        result = {
            **music_features,
            **speech_data,
            "audio_type": self._classify_audio_type(music_features, speech_data)
        }
        
        return result
    
    def _analyze_music_features(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze music features: BPM, beats, energy, etc.
        """
        try:
            # Load audio
            y, sr = librosa.load(video_path, sr=22050)
            duration = len(y) / sr
            
            # BPM and beat detection
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            bpm = float(tempo) if not isinstance(tempo, np.ndarray) else float(tempo[0])
            beat_times = librosa.frames_to_time(beats, sr=sr).tolist()
            
            # Find the first strong beat (beat drop)
            beat_drop = beat_times[0] if beat_times else 1.0
            
            # Energy analysis (RMS energy over time)
            rms = librosa.feature.rms(y=y)[0]
            energy_times = librosa.frames_to_time(range(len(rms)), sr=sr)
            
            # Find energy peaks (impact moments)
            energy_peaks = self._find_energy_peaks(rms, energy_times)
            
            # Spectral features for music detection
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
            
            # Music detection heuristic
            has_music = self._detect_music(bpm, spectral_flatness, rms)
            
            # Create energy timeline (normalized)
            energy_timeline = self._create_energy_timeline(rms, energy_times, duration)
            
            return {
                "duration": round(duration, 2),
                "bpm": round(bpm, 1),
                "beat_drop": round(beat_drop, 2),
                "beat_times": [round(t, 2) for t in beat_times],
                "has_music": has_music,
                "energy_peaks": energy_peaks,
                "energy_timeline": energy_timeline,
                "sync_points": self._extract_sync_points(beat_times, energy_peaks)
            }
            
        except Exception as e:
            print(f"  ⚠ Music analysis failed: {e}")
            return {
                "duration": 0,
                "bpm": 120.0,
                "beat_drop": 1.0,
                "beat_times": [],
                "has_music": True,
                "energy_peaks": [],
                "energy_timeline": [],
                "sync_points": []
            }
    
    def _detect_music(
        self,
        bpm: float,
        spectral_flatness: np.ndarray,
        rms: np.ndarray
    ) -> bool:
        """
        Detect if audio contains music vs pure speech/silence.
        """
        # Music typically has:
        # - Consistent BPM in 60-180 range
        # - Lower spectral flatness (more tonal)
        # - Consistent energy levels
        
        has_consistent_bpm = 60 <= bpm <= 180
        avg_flatness = np.mean(spectral_flatness)
        energy_variance = np.std(rms) / (np.mean(rms) + 1e-6)
        
        # Music is tonal (low flatness) with moderate energy consistency
        is_tonal = avg_flatness < 0.3
        has_rhythm = has_consistent_bpm
        
        return has_rhythm and is_tonal
    
    def _find_energy_peaks(
        self,
        rms: np.ndarray,
        energy_times: np.ndarray,
        min_prominence: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Find significant energy peaks (impact moments, drops, etc.)
        """
        from scipy.signal import find_peaks
        
        # Normalize RMS
        rms_normalized = rms / (np.max(rms) + 1e-6)
        
        # Find peaks
        peaks, properties = find_peaks(
            rms_normalized,
            prominence=min_prominence,
            distance=10  # Minimum distance between peaks
        )
        
        energy_peaks = []
        for peak_idx in peaks[:10]:  # Limit to top 10 peaks
            if peak_idx < len(energy_times):
                energy_peaks.append({
                    "time": round(float(energy_times[peak_idx]), 2),
                    "intensity": round(float(rms_normalized[peak_idx]), 2)
                })
        
        return sorted(energy_peaks, key=lambda x: x["intensity"], reverse=True)
    
    def _create_energy_timeline(
        self,
        rms: np.ndarray,
        energy_times: np.ndarray,
        duration: float,
        num_points: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Create a simplified energy timeline with evenly spaced points.
        """
        rms_normalized = rms / (np.max(rms) + 1e-6)
        
        timeline = []
        for i in range(num_points):
            t = (i / num_points) * duration
            # Find closest energy value
            idx = int((i / num_points) * len(rms))
            idx = min(idx, len(rms) - 1)
            
            timeline.append({
                "time": round(t, 2),
                "energy": round(float(rms_normalized[idx]), 2)
            })
        
        return timeline
    
    def _extract_sync_points(
        self,
        beat_times: List[float],
        energy_peaks: List[Dict[str, Any]]
    ) -> List[float]:
        """
        Extract key sync points for video transitions.
        Combines beats and energy peaks.
        """
        sync_points = set()
        
        # Add first few beats
        for beat in beat_times[:8]:
            sync_points.add(round(beat, 2))
        
        # Add top energy peaks
        for peak in energy_peaks[:5]:
            sync_points.add(peak["time"])
        
        return sorted(list(sync_points))
    
    def _transcribe_with_whisper(self, video_path: str) -> Dict[str, Any]:
        """
        Transcribe speech using OpenAI Whisper API.
        Returns transcript with timestamps.
        """
        try:
            # Extract audio to temporary file
            temp_audio_path = self._extract_audio_for_whisper(video_path)
            
            if not temp_audio_path or not os.path.exists(temp_audio_path):
                return {"has_speech": False, "transcript": None, "speech_segments": []}
            
            # Call Whisper API
            with open(temp_audio_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model=self.whisper_model,
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
            
            # Clean up temp file
            try:
                os.remove(temp_audio_path)
            except:
                pass
            
            # Parse response
            transcript = response.text.strip() if hasattr(response, 'text') else ""
            segments = []
            
            if hasattr(response, 'segments') and response.segments:
                for seg in response.segments:
                    segments.append({
                        "start": round(seg.start, 2),
                        "end": round(seg.end, 2),
                        "text": seg.text.strip()
                    })
            
            has_speech = len(transcript) > 10  # More than 10 chars = has speech
            
            return {
                "has_speech": has_speech,
                "transcript": transcript if has_speech else None,
                "speech_segments": segments
            }
            
        except Exception as e:
            print(f"  ⚠ Whisper transcription failed: {e}")
            return {"has_speech": False, "transcript": None, "speech_segments": []}
    
    def _extract_audio_for_whisper(self, video_path: str) -> Optional[str]:
        """
        Extract audio from video to a temporary file for Whisper.
        Whisper requires audio file format (mp3, wav, etc.)
        """
        try:
            from moviepy.editor import VideoFileClip
            
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            ext = "wav" if self.whisper_format.lower() == "wav" else "mp3"
            temp_path = os.path.join(temp_dir, f"whisper_temp.{ext}")
            
            video = VideoFileClip(video_path)
            if video.audio is not None:
                video.audio.write_audiofile(
                    temp_path,
                    fps=self.whisper_sample_rate,
                    codec="pcm_s16le" if ext == "wav" else "libmp3lame",
                    verbose=False,
                    logger=None
                )
                video.close()
                return temp_path
            else:
                video.close()
                return None
                
        except Exception as e:
            print(f"  ⚠ Audio extraction for Whisper failed: {e}")
            return None
    
    def _classify_audio_type(
        self,
        music_features: Dict[str, Any],
        speech_data: Dict[str, Any]
    ) -> str:
        """
        Classify the overall audio type.
        """
        has_music = music_features.get("has_music", False)
        has_speech = speech_data.get("has_speech", False)
        
        if has_speech and has_music:
            return "speech_over_music"  # Voiceover with background music
        elif has_speech:
            return "speech_only"  # Talking head, dialogue
        elif has_music:
            return "music_only"  # Music-driven content
        else:
            return "ambient"  # Sound effects, ambient noise
    
    def get_speech_at_timestamp(
        self,
        speech_segments: List[Dict[str, Any]],
        timestamp: float,
        window: float = 1.0
    ) -> Optional[str]:
        """
        Get speech text near a specific timestamp.
        
        Args:
            speech_segments: List of speech segments with start, end, text
            timestamp: Target timestamp in seconds
            window: Time window to search around timestamp
        
        Returns:
            Text spoken near the timestamp, or None
        """
        for segment in speech_segments:
            seg_start = segment.get("start", 0)
            seg_end = segment.get("end", 0)
            
            # Check if timestamp falls within or near this segment
            if (seg_start - window <= timestamp <= seg_end + window):
                return segment.get("text", "")
        
        return None
    
    def save_analysis(self, analysis: Dict[str, Any], output_path: str):
        """Save audio analysis to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)


if __name__ == "__main__":
    # Test the audio analyzer
    analyzer = AudioAnalyzer()
    # analysis = analyzer.analyze_audio("test_video.mp4")
    # print(json.dumps(analysis, indent=2))
