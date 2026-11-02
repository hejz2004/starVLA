"""Short video names and a caption above the untouched dual-camera frames."""
from pathlib import Path
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def video_filename(category, success):
    category = re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")
    return f"{category}_{'suc' if success else 'fail'}.mp4"


def video_header(width, job, instruction, success):
    font = ImageFont.load_default(size=14)
    title_font = ImageFont.load_default(size=16)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines, line = [], ""
    # Wrap by measured pixels, including long words and explicit newlines.
    for char in "Instruction: " + instruction:
        if char == "\n" or (line and measure.textlength(line + char, font=font) > width - 20):
            lines.append(line.rstrip()); line = ""
        if char != "\n": line += char
    if line: lines.append(line.rstrip())
    height = ((62 + 19 * len(lines) + 15) // 16) * 16
    header = Image.new("RGB", (width, height), (20, 24, 32))
    draw = ImageDraw.Draw(header)
    suite = {"libero_10": "Long", "libero_goal": "Goal", "libero_object": "Object", "libero_spatial": "Spatial"}[job["suite"]]
    status = "SUC" if success else "FAIL"
    draw.text((10, 8), f"Task: {suite} / {job['task_id']}   |   {status}", font=title_font, fill=(112, 228, 168) if success else (255, 180, 130))
    draw.text((10, 32), "Perturbation: " + job["category"], font=font, fill=(180, 205, 240))
    for i, line in enumerate(lines):
        draw.text((10, 54 + i * 19), line, font=font, fill=(245, 245, 245))
    return np.asarray(header)


def save_rollout_video(path, frames, *, job, instruction):
    import imageio.v2 as imageio
    if not frames: raise ValueError("Empty rollout video")
    # The ordinary loop passes a descriptive name; never create that long name.
    path = Path(path)
    success = path.stem.endswith("_success")
    path = path.parent / job["key"] / video_filename(job["category"], success)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.mp4")
    header = video_header(frames[0].shape[1], job, instruction, success)
    try:
        with imageio.get_writer(temporary, fps=10, ffmpeg_params=["-threads", "1"]) as writer:
            for frame in frames:
                writer.append_data(np.concatenate((header, frame), axis=0))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
