'''\
Animated Subtitles Generator
This script processes a given video and its subtitle file to render animated subtitles with optional animations overlayed on the video.
'''

# Import necessary libraries for video processing, subtitle parsing, and image rendering
from moviepy.editor import VideoFileClip, VideoClip, CompositeVideoClip, CompositeAudioClip
import pysrt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from animations import *  # Contains animation classes such as PushAndGrowAnimation
import argparse
import os
import multiprocessing
from tqdm import tqdm
from functools import partial


class AnimatedSubtitleGenerator:
    """
    A generator class for producing animated subtitles overlay for videos.
    
    This class loads a video and its corresponding subtitle file and renders animated subtitles 
    using PIL and moviepy. It supports various graphical options such as font, stroke, caption positioning, and animations.
    """
    def __init__(self, video_path, srt_path, font_path='./fonts/moon_get-Heavy.ttf', font_size=25,
                 caption_position='bottom', caption_padding=0, animation=None, window_size=3, spacing=10,
                 all_caps=False, highlight=True, stroke_width=2, stroke_color='black'):
        """
        Initialize the subtitle generator with video and subtitle files along with display settings.
        
        Parameters:
            video_path (str): Path to the video file.
            srt_path (str): Path to the SRT subtitle file.
            font_path (str): Path to the font file for subtitles.
            font_size (int): Font size for rendering subtitles.
            caption_position (str or tuple): Position of the captions ('center', 'top', 'bottom' or coordinate tuple).
            caption_padding (int): Padding for caption positioning when using 'top' or 'bottom'.
            animation: Optional animation object to apply effects (e.g., PushAndGrowAnimation).
            window_size (int): Number of words shown in the subtitle window.
            spacing (int): Spacing between words.
            all_caps (bool): If True, converts subtitles to all capital letters.
            highlight (bool): If True, highlights the currently active word.
            stroke_width (int): Stroke thickness for the text outline.
            stroke_color (str): Color for the text stroke.
        """
        self.video_path = video_path
        self.srt_path = srt_path
        self.video = VideoFileClip(video_path, audio=True)
        self.subs = pysrt.open(srt_path)
        self.transcript = self._build_transcript()
        
        # Try loading the specified font, or fall back to default if it fails.
        try:
            self.font = ImageFont.truetype(font_path, font_size)
        except Exception:
            self.font = ImageFont.load_default()
        
        self.caption_position = caption_position
        # Validate the caption_position parameter.
        if isinstance(self.caption_position, str):
            if self.caption_position not in ['center', 'top', 'bottom']:
                raise ValueError("Invalid caption position string. Allowed values: 'center', 'top', 'bottom'.")
        elif isinstance(self.caption_position, (tuple, list)):
            if len(self.caption_position) != 2 or not all(isinstance(num, int) for num in self.caption_position):
                raise ValueError("Caption position must be a tuple of two integers.")
        else:
            raise ValueError("Caption position must be either a string ('center', 'top', 'bottom') or a tuple of two integers.")
        
        self.caption_padding = caption_padding
        self.window_size = window_size
        self.spacing = spacing
        self.all_caps = all_caps
        self.highlight = highlight
        self.animation = animation
        self.stroke_width = stroke_width
        self.stroke_color = stroke_color
        
        # Caches for word size and layout measurements
        self._word_size_cache = {}
        self._layout_cache = {}

        # Gap between words adjusted if all_caps is True
        self.effective_gap = self.spacing * (1.5 if self.all_caps else 1)

        # Compute base canvas dimensions and then fixed dimensions for animations
        self.base_canvas_size = self._compute_base_dimensions()
        self.fixed_canvas_size = self.compute_fixed_dimensions()

    def _build_transcript(self):
        """
        Build a transcript list from the SRT file.
        
        Returns:
            list: Each element is a dictionary with 'word', 'start', and 'end' time (in seconds).
        """
        transcript = []
        for sub in self.subs:
            transcript.append({
                'word': sub.text.strip(),
                'start': sub.start.ordinal / 1000,  # Convert ms to seconds
                'end': sub.end.ordinal / 1000
            })
        return transcript

    def _get_word_size(self, word, add_stroke=True):
        """
        Calculate the drawing size of a word with the selected font.
        
        Uses caching to avoid recalculating the dimensions every time.
        
        Parameters:
            word (str): The word to be measured.
            add_stroke (bool): Whether to add extra space for stroke.
        
        Returns:
            tuple: (width, height) of the word in pixels.
        """
        key = word.upper() if self.all_caps else word
        cache_key = (key, add_stroke)
        if cache_key in self._word_size_cache:
            return self._word_size_cache[cache_key]
        
        if hasattr(self.font, 'getbbox'):
            bbox = self.font.getbbox(key)
            if add_stroke:
                stroke_adjust = self.stroke_width * 2
                w = (bbox[2] - bbox[0]) + stroke_adjust
                h = (bbox[3] - bbox[1]) + stroke_adjust
            else:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
        else:
            dummy_img = Image.new('RGBA', (1, 1))
            draw = ImageDraw.Draw(dummy_img)
            w, h = draw.textsize(key, font=self.font)
            if add_stroke:
                stroke_adjust = self.stroke_width * 2
                w += stroke_adjust
                h += stroke_adjust
        
        self._word_size_cache[cache_key] = (w, h)
        return (w, h)

    def _compute_base_dimensions(self):
        """
        Compute the base canvas dimensions needed to display a block of subtitle words (without animation scaling).
        
        Returns:
            tuple: (width, height) of the base canvas.
        """
        effective_gap = self.spacing * (1.5 if self.all_caps else 1)
        
        if not self.transcript:
            return (200, 50)
        
        word_widths = []
        max_height = 0
        
        # Measure each word's dimensions
        for entry in self.transcript:
            word = entry['word']
            w, h = self._get_word_size(word, add_stroke=True)
            word_widths.append(w)
            max_height = max(max_height, h)
        
        effective_window = min(self.window_size, len(word_widths))
        if effective_window == 0:
            total_width = 0
        elif effective_window == 1:
            total_width = max(word_widths)
        else:
            max_total = 0
            # Determine maximum width for any continuous block of words
            for i in range(len(word_widths) - effective_window + 1):
                window_width = sum(word_widths[i:i+effective_window]) + effective_gap * (effective_window - 1)
                max_total = max(max_total, window_width)
            total_width = max_total
        
        # Padding around the text
        horizontal_padding = 40
        vertical_padding = 40
        if isinstance(self.caption_position, str) and self.caption_position in ['top', 'bottom']:
            vertical_padding = self.caption_padding
        
        return (int(total_width + horizontal_padding), int(max_height + vertical_padding))

    def compute_fixed_dimensions(self):
        """
        Compute the fixed canvas dimensions, adjusting for animation scaling to avoid clipping.
        
        Returns:
            tuple: (width, height) of the fixed canvas.
        """
        base_width, base_height = self._compute_base_dimensions()
        if self.animation and hasattr(self.animation, 'target_scale'):
            # Increase buffer to allow growth space during animation
            scale = self.animation.target_scale * 3.0
            return (int(base_width * scale), int(base_height * scale))
        else:
            return (base_width, base_height)

    def get_current_subtitle(self, t):
        """
        Retrieve the current subtitle entry based on time t.
        
        Parameters:
            t (float): Current time in seconds.
        
        Returns:
            dict: The subtitle entry corresponding to the given time.
        """
        for entry in self.transcript:
            if entry['start'] <= t < entry['end']:
                return entry
        # If no active subtitle, return the last entry that ended before t, or default
        for entry in reversed(self.transcript):
            if entry['end'] <= t:
                return entry
        return self.transcript[0] if self.transcript else {'word': '', 'start': 0, 'end': 0}

    def get_current_subtitle_index(self, t):
        """
        Get the index of the current subtitle in the transcript based on time t.
        
        Parameters:
            t (float): Current time in seconds.
        
        Returns:
            int: Index of the active subtitle entry.
        """
        for i, entry in enumerate(self.transcript):
            if entry['start'] <= t < entry['end']:
                return i
        for i in range(len(self.transcript) - 1, -1, -1):
            if self.transcript[i]['end'] <= t:
                return i
        return 0

    def make_text_frame_rgba(self, t):
        """
        Create an RGBA image frame with the rendered subtitle text at time t.
        
        Parameters:
            t (float): Current time in seconds.
        
        Returns:
            numpy.ndarray: An RGBA image frame as a numpy array.
        """
        if not self.transcript:
            img = Image.new('RGBA', self.fixed_canvas_size, (0, 0, 0, 0))
            return np.array(img)
        
        current_index = self.get_current_subtitle_index(t)
        block_index = current_index // self.window_size
        start_index = block_index * self.window_size
        end_index = start_index + self.window_size
        window_entries = self.transcript[start_index:end_index]
        
        # Pad the window if it's incomplete
        if len(window_entries) < self.window_size:
            pad_needed = self.window_size - len(window_entries)
            window_entries = window_entries + ([{'word': ''}] * pad_needed)
        
        active_index_in_block = current_index % self.window_size
        
        # Reuse layout if already computed
        if block_index in self._layout_cache:
            widths, heights, total_window_width, window_height, x_start, y = self._layout_cache[block_index]
        else:
            widths = []
            heights = []
            for entry in window_entries:
                word = entry.get('word', '')
                w, h = self._get_word_size(word, add_stroke=False)
                widths.append(w)
                heights.append(h)
            window_height = max(heights) if heights else 0
            total_window_width = sum(widths) + self.effective_gap * (len(widths) - 1) if widths else 0
            
            if isinstance(self.caption_position, str):
                if self.caption_position == 'top':
                    y = self.caption_padding
                elif self.caption_position == 'bottom':
                    y = self.fixed_canvas_size[1] - window_height - self.caption_padding
                else:  # center
                    y = (self.fixed_canvas_size[1] - window_height) / 2 + self.caption_padding
                x_start = (self.fixed_canvas_size[0] - total_window_width) / 2
            else:
                if self.animation and hasattr(self.animation, 'target_scale'):
                    base_width, base_height = self.base_canvas_size
                    offset_x = (self.fixed_canvas_size[0] - base_width) / 2
                    offset_y = (self.fixed_canvas_size[1] - base_height) / 2
                else:
                    offset_x, offset_y = 0, 0
                x_start = offset_x + (self.base_canvas_size[0] - total_window_width) / 2
                y = offset_y + (self.fixed_canvas_size[1] - window_height) / 2
            self._layout_cache[block_index] = (widths, heights, total_window_width, window_height, x_start, y)
        
        # Create a transparent image and draw the text
        img = Image.new('RGBA', self.fixed_canvas_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        current_x = x_start
        for idx, entry in enumerate(window_entries):
            word = entry.get('word', '')
            if self.all_caps:
                word = word.upper()
            # Highlight active word if enabled
            fill_color = 'yellow' if (self.highlight and idx == active_index_in_block) else 'white'
            
            # Determine current scale from animation if applicable
            if self.animation:
                if hasattr(self.animation, 'get_scale'):
                    current_scale = self.animation.get_scale(t)
                elif hasattr(self.animation, 'target_scale'):
                    current_scale = self.animation.target_scale
                else:
                    current_scale = 1
            else:
                current_scale = 1

            adjusted_stroke_width = int(round(self.stroke_width * current_scale))
            draw.text((current_x, y), word, font=self.font, fill=fill_color,
                      stroke_width=adjusted_stroke_width, stroke_fill=self.stroke_color)
            current_x += widths[idx] + self.effective_gap
        
        return np.array(img)

    def make_text_frame_rgb(self, t):
        """
        Generate an RGB frame by stripping the alpha channel from the RGBA frame at time t.
        
        Parameters:
            t (float): Current time in seconds.
        
        Returns:
            numpy.ndarray: An RGB image frame.
        """
        rgba = self.make_text_frame_rgba(t)
        return rgba[:, :, :3]

    def make_text_mask(self, t):
        """
        Create a normalized alpha mask for the subtitle text at time t.
        
        Parameters:
            t (float): Current time in seconds.
        
        Returns:
            numpy.ndarray: A mask with values between 0 and 1.
        """
        rgba = self.make_text_frame_rgba(t)
        alpha = rgba[:, :, 3] / 255.0
        return alpha

    def animated_text_frame_rgb(self, t):
        """
        Generate an animated RGB frame with applied animation effects at time t.
        
        Parameters:
            t (float): Current time in seconds.
        
        Returns:
            numpy.ndarray: An animated RGB frame.
        """
        frame = self.make_text_frame_rgba(t)
        if self.animation:
            active_sub = self.get_current_subtitle(t)
            frame = self.animation.apply(frame, t, active_sub)
        return frame[:, :, :3]

    def animated_text_mask(self, t):
        """
        Generate an animated alpha mask with applied animation effects at time t.
        
        Parameters:
            t (float): Current time in seconds.
        
        Returns:
            numpy.ndarray: An animated mask with values between 0 and 1.
        """
        frame = self.make_text_frame_rgba(t)
        if self.animation:
            active_sub = self.get_current_subtitle(t)
            frame = self.animation.apply(frame, t, active_sub)
        alpha = frame[:, :, 3] / 255.0
        return alpha

    def create_text_clip(self):
        """
        Create a moviepy VideoClip for the subtitle text along with its mask.
        
        Returns:
            VideoClip: A clip containing the animated subtitle text and transparency mask.
        """
        text_clip = VideoClip(make_frame=self.animated_text_frame_rgb, duration=self.video.duration)
        mask_clip = VideoClip(make_frame=self.animated_text_mask, duration=self.video.duration)
        mask_clip.ismask = True  # Designate this clip as a mask
        text_clip = text_clip.set_mask(mask_clip)
        text_clip = text_clip.set_position(self.caption_position)
        return text_clip

    def create_composite_video(self, output_path):
        """
        Create the final composite video by overlaying animated subtitles on the original video.
        
        Parameters:
            output_path (str): Path to save the output video file.
        """
        text_clip = self.create_text_clip().set_duration(self.video.duration)
        audio_clip = CompositeAudioClip([self.video.audio]).set_duration(self.video.duration)
        final_clip = CompositeVideoClip([self.video, text_clip]).set_audio(audio_clip)
        final_clip.write_videofile(output_path, codec='libx264', audio=True, audio_codec='aac', audio_bitrate='192k')


# Add new helper function before the AnimatedSubtitleGenerator class
def process_single_video(args_tuple, base_args):
    """Worker function for processing individual videos with progress reporting"""
    video_path, srt_path, output_path = args_tuple
    try:
        animation = None
        if base_args.animation.lower() == 'pushandgrow':
            from animations import PushAndGrowAnimation
            animation = PushAndGrowAnimation(
                target_scale=base_args.target_scale,
                speed=base_args.speed,
                push_strength=base_args.push_strength
            )
            
        generator = AnimatedSubtitleGenerator(
            video_path,
            srt_path,
            font_path=base_args.font_path,
            font_size=base_args.font_size,
            caption_position=base_args.caption_position,
            caption_padding=base_args.caption_padding,
            window_size=base_args.words_on_screen,
            spacing=base_args.spacing,
            all_caps=base_args.all_caps,
            highlight=base_args.highlight,
            animation=animation,
            stroke_width=base_args.stroke_width,
            stroke_color=base_args.stroke_color
        )
        generator.create_composite_video(output_path)
        return (True, os.path.basename(video_path), None)
    except Exception as e:
        return (False, os.path.basename(video_path), str(e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Animated Subtitle Generator Command Line Utility')
    parser.add_argument('video_path', help='Path to the input video file or directory containing videos')
    parser.add_argument('srt_path', help='Path to the subtitle file (.srt) or directory containing subtitle files. If a directory is given, it will be searched for matching srt files.')
    parser.add_argument('output_path', help='Path to the output video file or directory for output files')
    parser.add_argument('--font-path', default='./fonts/moon_get-Heavy.ttf', help='Path to the font file')
    parser.add_argument('--font-size', type=int, default=25, help='Font size')
    parser.add_argument('--caption-position', type=str, default='bottom', choices=['center', 'top', 'bottom'], help='Caption position')
    parser.add_argument('--words-on-screen', type=int, default=4, help='Number of words to display on screen')
    parser.add_argument('--spacing', type=int, default=10, help='Spacing between words')
    parser.add_argument('--all-caps', action='store_true', help='Convert subtitles to all caps')
    parser.add_argument('--no-highlight', dest='highlight', action='store_false', help='Disable word highlighting')
    parser.add_argument('--animation', type=str, default='pushandgrow', choices=['none', 'pushandgrow'], help='Type of animation effect')
    parser.add_argument('--target-scale', type=float, default=1.2, help='Target scale for the animation')
    parser.add_argument('--speed', type=float, default=5.0, help='Speed for the animation')
    parser.add_argument('--push-strength', type=float, default=0.5, help='Push strength for the animation')
    parser.add_argument('--stroke-width', type=int, default=2, help='Thickness of the stroke outline for subtitle text')
    parser.add_argument('--stroke-color', default='black', help='Color of the stroke outline for subtitle text')
    parser.add_argument('--caption-padding', type=int, default=0, help='Vertical padding offset for top or bottom captions')
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(), 
                      help='Number of parallel workers to use for processing')
    args = parser.parse_args()
    
    # Batch processing if video_path is a directory
    if os.path.isdir(args.video_path):
        # Ensure output_path is a directory; create if it doesn't exist
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
            print(f"Created output directory: {args.output_path}")
        elif not os.path.isdir(args.output_path):
            print('Error: output_path exists and is not a directory.')
            exit(1)
        
        # Determine the directory containing subtitle files
        srt_dir = args.srt_path if os.path.isdir(args.srt_path) else args.video_path
        
        # Get list of video files (skip hidden files)
        video_files = [fname for fname in os.listdir(args.video_path) 
                      if fname.lower().endswith('.mp4') and not fname.startswith('.')]
        
        # Prepare task list
        tasks = []
        for video_fname in video_files:
            video_full_path = os.path.join(args.video_path, video_fname)
            base = os.path.splitext(video_fname)[0]
            srt_file = os.path.join(srt_dir, base + '.srt')
            if not os.path.exists(srt_file):
                print(f'Subtitle file not found for {video_fname} at expected location {srt_file}, skipping.')
                continue
            output_file = os.path.join(args.output_path, base + '_output.mp4')
            tasks.append((video_full_path, srt_file, output_file))
        
        # Process in parallel with progress bar
        processed_videos = []
        failed_videos = []
        
        with multiprocessing.Pool(processes=args.workers) as pool:
            worker_func = partial(process_single_video, base_args=args)
            results = list(tqdm(pool.imap(worker_func, tasks), 
                             total=len(tasks),
                             desc='Processing videos',
                             unit='vid'))
            
            for success, fname, error in results:
                if success:
                    processed_videos.append(fname)
                else:
                    failed_videos.append((fname, error))
        
        # Report results
        if processed_videos:
            print("\n\033[92mSuccessfully processed:")
            for f in processed_videos:
                print(f"  ✓ {f}")
        if failed_videos:
            print("\n\033[91mFailed to process:")
            for f, e in failed_videos:
                print(f"  ✗ {f} - {e}")
        if processed_videos:
            print("\n\033[93mSkipped due to missing subtitles:")
            for f in processed_videos:
                print(f"  ! {f}")
        print("\033[0m")  # Reset colors
    else:
        # Single file processing; ensure output file has proper extension
        if not os.path.splitext(args.output_path)[1]:
            args.output_path = args.output_path + ".mp4"
            print(f"No output file extension detected. Using: {args.output_path}")
        animation = None
        if args.animation.lower() == 'pushandgrow':
            from animations import PushAndGrowAnimation
            animation = PushAndGrowAnimation(target_scale=args.target_scale, speed=args.speed, push_strength=args.push_strength)
        
        generator = AnimatedSubtitleGenerator(
            args.video_path,
            args.srt_path,
            font_path=args.font_path,
            font_size=args.font_size,
            caption_position=args.caption_position,
            caption_padding=args.caption_padding,
            window_size=args.words_on_screen,
            spacing=args.spacing,
            all_caps=args.all_caps,
            highlight=args.highlight,
            animation=animation,
            stroke_width=args.stroke_width,
            stroke_color=args.stroke_color
        )
        generator.create_composite_video(args.output_path)