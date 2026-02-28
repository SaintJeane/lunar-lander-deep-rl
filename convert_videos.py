import os
import glob
from moviepy import VideoFileClip

def convert_videos_to_gifs(base_dir="./videos", output_dir="./assets/videos"):
    """
    Find all mp4 files in subdirectories of base_dir and
    converts them to optimized GIFs in output_dir.
    """
    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all mp4 files in any subfolder of videos/
    search_path = os.path.join(base_dir, "**", "*.mp4")
    video_files = glob.glob(search_path, recursive=True)
    
    if not video_files:
        print(f"❌ No .mp4 files found in {base_dir}")
        return
    
    print(f"🔍 Found {len(video_files)} videos. Starting conversion...")
    
    for video_path in video_files:
        # Generate a unique name for the GIF based on folder and filename
        # Example: videos/dqn_2026.../rl-video-episode-0.mp4 -> dqn_2026..._episode-0.gif
        folder_name = os.path.basename(os.path.dirname(video_path))
        file_name = os.path.basename(video_path).replace(".mp4", "")
        gif_name = f"{folder_name}_{file_name}.gif"
        output_path = os.path.join(output_dir, gif_name)
        
        print(f"🎬 Converting: {folder_name}/{file_name}.mp4 -> {gif_name}")
        
        try:
            # Load the video
            clip = VideoFileClip(video_path)
            
            # Optimization: Resize to a smaller width for faster loading in README
            # and set frame rate lower (15 fps) to reduce file size
            clip = clip.resized(width=480)
            
            # Write the GIF
            clip.write_gif(output_path, fps=15)
            clip.close()
        except Exception as e:
            print(f"❌ Error converting {video_path}: {e}")
            
    print(f"\n✅ All done! GIFs saved in: {output_dir}")
    
if __name__ == "__main__":
    convert_videos_to_gifs() 
    
# run python convert_videos.py