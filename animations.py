from PIL import Image
import numpy as np
import math
from PIL import ImageDraw

# New animation: PushAndGrowAnimation
class PushAndGrowAnimation:
    def __init__(self, target_scale=1.2, speed=1.0, push_strength=1.0):
        self.target_scale = target_scale
        self.speed = speed
        self.push_strength = push_strength

    def apply(self, frame, t, active_sub):
        """Apply the push-and-grow animation effect on the provided frame at time t using the active subtitle.
        
        This function performs the following steps:
          1. Ensures the input frame is in RGBA format.
          2. Detects the active subtitle region via a yellow color mask.
          3. Computes the animation progress based on the current time and the subtitle's time bounds.
          4. Scales the active region and calculates a push offset based on the easing function.
          5. Composites the final frame by blending the shifted background and the scaled active region.
        
        Args:
          frame (numpy.ndarray): The input video frame as a NumPy array.
          t (float): The current time in the video.
          active_sub (dict): A dictionary containing the keys 'start' and 'end' for the active subtitle.
        
        Returns:
          numpy.ndarray: The animated frame as a NumPy array in RGBA format.
        """
        # 1. Ensure the frame is in RGBA format
        if frame.shape[2] == 4:
            base = Image.fromarray(frame, 'RGBA')
            rgba = frame
        else:
            base = Image.fromarray(frame).convert("RGBA")
            rgba = np.array(base)
        W, H = base.size

        # 2. Create a yellow mask to detect the active subtitle text
        yellow_mask = (rgba[:, :, 0] > 200) & (rgba[:, :, 1] > 200) & (rgba[:, :, 2] < 150) & (rgba[:, :, 3] > 0)
        if not yellow_mask.any():
            return frame  # No active subtitle detected, return original frame.

        # 3. Determine the bounding box of the active subtitle region
        ys, xs = np.nonzero(yellow_mask)  # Get indices where yellow_mask is True
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        # Expand the bounding box slightly by 5 pixels for margin
        x0 = max(0, x_min - 5)
        x1 = min(W, x_max + 5)
        y0 = max(0, y_min - 5)
        y1 = min(H, y_max + 5)
        active_region = base.crop((x0, y0, x1, y1))
        orig_width, orig_height = active_region.size

        # 4. Compute the animation progress based on the active subtitle's start and end times
        active_start = active_sub.get('start', t)
        active_end = active_sub.get('end', t + 1e-9)  # Use a small epsilon to avoid division by zero
        progress = np.clip((t - active_start) * self.speed / (active_end - active_start), 0, 1)

        # 5. Calculate the scaling factor and resize the active region
        scale_delta = self.target_scale - 1
        scale_factor = 1 + scale_delta * progress
        new_width = int(orig_width * scale_factor)
        new_height = int(orig_height * scale_factor)
        scaled_active = active_region.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # 6. Center the scaled active region within the original bounding box
        active_center_x = (x0 + x1) // 2
        active_center_y = (y0 + y1) // 2
        new_x = max(0, min(active_center_x - new_width // 2, W - new_width))
        new_y = max(0, min(active_center_y - new_height // 2, H - new_height))

        # 7. Compute the push offset using an ease-out cubic function
        extra_width = new_width - orig_width
        push_progress = 1 - (1 - progress) ** 3
        push_offset = int(extra_width * push_progress * self.push_strength)

        # 8. Composite the final image by shifting background sections and overlaying the scaled active region
        bg = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        # Shift left background if applicable
        if x0 > 0:
            left_bg = base.crop((0, 0, x0, H))
            bg.paste(left_bg, (-push_offset, 0), left_bg)
        # Shift right background if applicable
        if x1 < W:
            right_bg = base.crop((x1, 0, W, H))
            bg.paste(right_bg, (x1 + push_offset, 0), right_bg)
        # Overlay the scaled active region
        bg.paste(scaled_active, (new_x, new_y), scaled_active)

        return np.array(bg) 