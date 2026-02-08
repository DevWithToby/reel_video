"""
Video Download Utility
Downloads videos from URLs (direct links, YouTube, Instagram, etc.)
"""
import os
import httpx
import yt_dlp
from typing import Optional
from urllib.parse import urlparse


class VideoDownloader:
    """Downloads videos from various sources"""
    
    def __init__(self, download_dir: str = "uploads"):
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)
    
    def download_video(self, url: str, output_filename: str) -> str:
        """
        Download video from URL.
        Supports:
        - Direct video links (.mp4, .mov, etc.)
        - YouTube URLs
        - Instagram Reels
        - TikTok videos
        - Other platforms supported by yt-dlp
        """
        output_path = os.path.join(self.download_dir, output_filename)
        
        # Check if it's a direct video link
        if self._is_direct_video_link(url):
            return self._download_direct_link(url, output_path)
        else:
            # Use yt-dlp for YouTube, Instagram, TikTok, etc.
            return self._download_with_ytdlp(url, output_path)
    
    def _is_direct_video_link(self, url: str) -> bool:
        """Check if URL is a direct video file link"""
        parsed = urlparse(url)
        path = parsed.path.lower()
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv']
        return any(path.endswith(ext) for ext in video_extensions)
    
    def _download_direct_link(self, url: str, output_path: str) -> str:
        """Download direct video link using httpx"""
        print(f"Downloading direct video link: {url}")
        with httpx.stream("GET", url, timeout=60.0, follow_redirects=True) as response:
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
        print(f"Downloaded to: {output_path}")
        return output_path
    
    def _download_with_ytdlp(self, url: str, output_path: str) -> str:
        """Download video using yt-dlp (YouTube, Instagram, TikTok, etc.)"""
        print(f"Downloading with yt-dlp: {url}")
        
        # Remove extension from output_path, yt-dlp will add it
        base_path = output_path.rsplit('.', 1)[0]
        
        ydl_opts = {
            # Better format selection to avoid SABR streaming issues
            # Try best video+audio, then best mp4, then best available
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best[ext=mp4]/best',
            'merge_output_format': 'mp4',  # Merge video+audio to mp4
            'outtmpl': base_path + '.%(ext)s',
            'quiet': False,
            'no_warnings': False,
            # Use different extractors to avoid SABR streaming
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'web'],  # Try android first, then web
                    'player_skip': ['webpage', 'configs'],  # Skip problematic parts
                }
            },
            # Retry on fragment errors
            'fragment_retries': 10,
            'retries': 10,
            # Use ffmpeg for merging (already installed)
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                
                # yt-dlp may change the extension, find the actual file
                for ext in ['mp4', 'webm', 'mkv', 'mov', 'm4a']:
                    possible_path = f"{base_path}.{ext}"
                    if os.path.exists(possible_path) and os.path.getsize(possible_path) > 0:
                        # If not mp4, check if conversion happened
                        if ext != 'mp4':
                            final_path = f"{base_path}.mp4"
                            # Check if conversion happened
                            if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                                print(f"Downloaded and converted to: {final_path}, size: {os.path.getsize(final_path)}")
                                return final_path
                            print(f"Downloaded as {ext}, file size: {os.path.getsize(possible_path)}")
                            return possible_path
                        file_size = os.path.getsize(possible_path)
                        print(f"Downloaded to: {possible_path}, size: {file_size}")
                        return possible_path
                
                # If we can't find the file, raise error
                raise ValueError(f"Downloaded file is empty or not found")
        except Exception as e:
            print(f"yt-dlp download failed: {e}")
            # Try fallback with simpler format
            print("Trying fallback format...")
            try:
                fallback_opts = {
                    'format': 'worst[ext=mp4]/worst',  # Use worst quality to avoid SABR
                    'outtmpl': base_path + '.%(ext)s',
                    'quiet': False,
                    'extractor_args': {
                        'youtube': {
                            'player_client': ['android'],  # Android client is more reliable
                        }
                    },
                    'fragment_retries': 10,
                    'retries': 10,
                }
                with yt_dlp.YoutubeDL(fallback_opts) as ydl:
                    ydl.download([url])
                    
                    for ext in ['mp4', 'webm', 'mkv', 'mov']:
                        possible_path = f"{base_path}.{ext}"
                        if os.path.exists(possible_path) and os.path.getsize(possible_path) > 0:
                            print(f"Downloaded with fallback to: {possible_path}, size: {os.path.getsize(possible_path)}")
                            return possible_path
                    
                    raise ValueError(f"Fallback download also failed - file is empty")
            except Exception as fallback_error:
                raise ValueError(f"Failed to download video from URL: {str(e)}. Fallback also failed: {str(fallback_error)}")

