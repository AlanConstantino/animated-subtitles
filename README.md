# Animated Subtitles Generator

Animated Subtitles Generator is a tool to overlay animated subtitles onto video files. This tool processes a given video and its corresponding subtitle file (in SRT format), and renders animated subtitles with customizable options such as font, position, and animation effects.

## Output

![Example 1](./videos/subtitle_demo.gif)

## Disclaimer

YOU NEED A WORD LEVEL TRANSCRIPTION SUBTITLE .SRT FILE TO USE THIS TOOL. THIS HAS NOT BEEN TESTED WITH SRT FILES THAT HAVE SUBTITLE TIMES THAT ARE NOT ON WORD BOUNDARIES.

This tool is a work in progress and is not yet ready for production use. It is a proof of concept and is not intended to be used as a production tool.

## Features

- Supports single file processing and batch processing (processing a directory of videos).
- Animated subtitle effects including a "push and grow" animation.
- Customization options for font, caption style, window size, spacing, and more.
- Generates output videos with the animated subtitles overlay applied.

## Requirements

- Python 3.6+
- [MoviePy](https://zulko.github.io/moviepy/) for video processing
- [pysrt](https://github.com/byroot/pysrt) for subtitle parsing
- [Pillow](https://python-pillow.org/) for image rendering
- [NumPy](https://numpy.org/) for numerical operations
- [tqdm](https://github.com/tqdm/tqdm) for progress display
- FFmpeg installed and available in your system PATH (required by MoviePy)

Install the Python package dependencies using pip:

```bash
pip install moviepy pysrt pillow numpy tqdm
```

## Usage

The script can be run via the command line. Its basic syntax is:

### Single File Processing

For processing a single video file with its corresponding SRT subtitle:

```bash
python animated_subtitles.py /path/to/video.mp4 /path/to/subtitle.srt /path/to/output.mp4 [options]
```

### Batch Processing

For processing all MP4 videos in a directory, with corresponding SRT files (matched by filename), and outputting to a specified directory:

```bash
python animated_subtitles.py /path/to/video_directory /path/to/subtitles_directory /path/to/output_directory [options]
```

### Positional Arguments

- `video_path`: Path to the input video file, or a directory containing multiple video files.
- `srt_path`: Path to the subtitle file (SRT format) for a single video, or a directory containing subtitle files. If a directory is given, the tool will search for matching SRT files based on the video filename.
- `output_path`: Path to the output video file, or directory where output videos will be stored.

### Optional Arguments

```
--font-path         Path to the font file used for rendering subtitles. Default: /Library/Fonts/moon_get-Heavy.ttf.
--font-size         Font size for subtitles. Default: 25.
--caption-position  Position of the subtitles on the video. Options: "center", "top", "bottom". Default: bottom.
--words-on-screen   Number of words to display on screen. Default: 4.
--spacing           Spacing between words. Default: 10.
--all-caps          Convert subtitles to all uppercase.
--no-highlight      Disable highlighting of the active word.
--animation         Animation effect to apply. Options: "pushandgrow", "none". Default: pushandgrow.
--target-scale      Scale factor for the animation effect. Default: 1.2.
--speed             Animation speed. Default: 5.0.
--push-strength     Strength of the push effect in the animation. Default: 0.5.
--stroke-width      Stroke thickness for subtitle text outline. Default: 2.
--stroke-color      Color of the stroke outline. Default: black.
--caption-padding   Vertical padding for top or bottom caption positions. Default: 0.
--workers           Number of parallel workers to use for batch processing. Default: number of CPU cores.
```

### Examples

1. **Single File Processing:**

   Process a single video with its corresponding SRT file:

   ```bash
   python animated_subtitles.py example_video.mp4 example_subtitles.srt example_video_output.mp4
   ```

2. **Batch Processing:**

   Process all MP4 videos in a directory, matching them with their respective SRT files (based on filename), and output the processed videos to an output directory:

   ```bash
   python animated_subtitles.py /path/to/video_directory /path/to/subtitles_directory /path/to/output_directory --workers 4
   ```

3. **Using Custom Font and Animation Settings:**

   ```bash
   python animated_subtitles.py video.mp4 subtitles.srt output.mp4 --font-path "path/to/custom_font.ttf" --font-size 30 --animation pushandgrow --target-scale 1.5 --speed 4.0 --push-strength 0.7
   ```

## How It Works

The tool reads the input video and its corresponding SRT subtitle file, builds a transcript, and renders animated subtitles onto each video frame. The "push and grow" animation effect highlights the active subtitle by scaling and shifting the subtitle region, creating a dynamic visual effect.

The video and audio are then combined to produce the final output video with the animated subtitles overlay.

## Troubleshooting

- Ensure that FFmpeg is installed and correctly added to your system's PATH.
- Verify that the input video and SRT files exist and are correctly specified.
- If custom fonts are used, confirm the font file path is correct.
- For issues related to subtitle parsing, ensure the subtitle file is in a valid SRT format.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- [MoviePy](https://zulko.github.io/moviepy/)
- [pillow](https://python-pillow.org/)
- [NumPy](https://numpy.org/)
- [pysrt](https://github.com/byroot/pysrt)
- [tqdm](https://github.com/tqdm/tqdm) 