"""ffprobe / ffmpeg 相關：讀取影片資訊、抽音訊、縮圖、轉相容播放檔。

ffmpeg、ffprobe 的位置在啟動時找好（config.TOOLS，找法見 config.find_tool）；
找不到時丟 config.ToolMissing，訊息可以直接給使用者看（API 回 503，任務顯示成失敗原因）。
"""
import json
import logging
import subprocess
from pathlib import Path
from typing import Callable

from . import config, safepath
from .config import SAMPLE_RATE, ToolMissing

log = logging.getLogger("media")

NO_WINDOW = subprocess.CREATE_NO_WINDOW

BROWSER_CONTAINERS = {".mp4", ".m4v", ".mov", ".webm"}
BROWSER_VCODECS = {"h264", "hevc", "vp8", "vp9", "av1"}
BROWSER_ACODECS = {"aac", "mp3", "opus", "vorbis", "flac"}
AUDIO_ONLY_EXTS = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac"}

# 相容播放檔的影像編碼：先用顯示卡的 NVENC，不能用時改用 CPU 的 libx264
NVENC_ARGS = ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "21"]
X264_ARGS = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"]
X264_NOTE = "顯示卡編碼不能用，改用 CPU 轉檔，會比較慢"
X264_FILE_NOTE = "這部影片顯示卡編碼不了（例如畫面太大），改用 CPU 轉檔，會比較慢"
# 這次執行已經確定 NVENC 不能用（驅動太舊、沒有 NVIDIA 顯示卡），之後的轉檔直接用 libx264。
# 只有某一部影片編不了（例如超過 NVENC 的尺寸上限）時不設，下一部照樣先用 NVENC
_nvenc_failed = {"value": False}


class FfmpegError(RuntimeError):
    """ffmpeg 執行失敗；stderr 是它最後的錯誤輸出。"""

    def __init__(self, message: str, stderr: str = ""):
        super().__init__(message)
        self.stderr = stderr


def probe(path: Path) -> dict:
    path = Path(path)
    ffprobe = config.require_tool("ffprobe")
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries",
             "format=duration:stream=codec_type,codec_name,width,height:stream_disposition=attached_pic",
             # 用完整路徑，檔名就算以 - 開頭也不會被 ffprobe 當成參數
             "-of", "json", str(path.absolute())],
            capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW,
        )
    except FileNotFoundError:
        raise ToolMissing("找不到 ffprobe，請重新解壓縮程式或執行 repair.bat")
    if out.returncode != 0:
        raise ValueError("無法讀取這個檔案，可能不是影片或檔案已損毀")
    data = json.loads(out.stdout or "{}")
    streams = data.get("streams", [])
    video = next(
        (s for s in streams if s.get("codec_type") == "video"
         and not s.get("disposition", {}).get("attached_pic")
         and s.get("codec_name") not in ("mjpeg", "png")),
        None,
    )
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise ValueError("這個檔案沒有音軌")
    ext = path.suffix.lower()
    vcodec = video.get("codec_name") if video else None
    acodec = audio.get("codec_name")
    if ext in AUDIO_ONLY_EXTS:
        playable = acodec in BROWSER_ACODECS
    else:
        playable = ext in BROWSER_CONTAINERS and vcodec in BROWSER_VCODECS and acodec in BROWSER_ACODECS
    return {
        "duration": float(data.get("format", {}).get("duration") or 0),
        "vcodec": vcodec,
        "acodec": acodec,
        "width": video.get("width") if video else None,
        "height": video.get("height") if video else None,
        "playable": 1 if playable else 0,
    }


def _run_with_progress(args: list, duration: float, on_progress: Callable[[float], None] | None,
                       check: Callable[[], None] | None):
    try:
        proc = subprocess.Popen(
            args + ["-progress", "pipe:1", "-nostats"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            creationflags=NO_WINDOW,
        )
    except FileNotFoundError:
        raise ToolMissing("找不到 ffmpeg，請重新解壓縮程式或執行 repair.bat")
    try:
        for line in proc.stdout:
            if check:
                check()
            if line.startswith("out_time_us=") and duration > 0 and on_progress:
                try:
                    on_progress(min(1.0, int(line.split("=", 1)[1]) / 1e6 / duration))
                except ValueError:
                    pass
        proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    if proc.returncode != 0:
        err = proc.stderr.read()[-800:]
        raise FfmpegError(f"ffmpeg 失敗：{err.strip()}", err)


def extract_audio(src: Path, dst: Path, duration: float, on_progress=None, check=None):
    ffmpeg = config.require_tool("ffmpeg")
    tmp = dst.with_suffix(".tmp.wav")
    try:
        _run_with_progress(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-map", "0:a:0", "-vn",
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(tmp)],
            duration, on_progress, check,
        )
    except BaseException:
        safepath.safe_unlink(tmp, dst.parent)  # 取消或失敗時不留半截的暫存檔
        raise
    tmp.replace(dst)


def make_thumbnail(src: Path, dst: Path, duration: float) -> bool:
    try:
        ffmpeg = config.require_tool("ffmpeg")
    except ToolMissing:
        return False
    at = min(max(duration * 0.15, 0), 90) if duration else 0
    try:
        res = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-ss", f"{at:.2f}", "-i", str(src), "-frames:v", "1",
             "-vf", "scale=480:-2", "-q:v", "4", str(dst)],
            capture_output=True, creationflags=NO_WINDOW,
        )
    except OSError:
        return False
    return res.returncode == 0 and dst.exists()


def _nvenc_problem(stderr: str) -> bool:
    """ffmpeg 的錯誤是不是 NVENC 本身不能用（沒有顯示卡、驅動版本不合、編碼器不存在），而不是來源檔的問題。"""
    text = (stderr or "").lower()
    return any(k in text for k in ("nvenc", "nvcuda", "cuda", "unknown encoder", "error while opening encoder",
                                   "no capable devices", "driver does not support"))


def nvenc_works(ffmpeg: str, timeout: float = 30) -> bool:
    """用 0.2 秒的純黑畫面試一次 NVENC（跟轉檔同樣的參數），能編就是顯示卡編碼本身沒問題。
    NVENC 轉某部影片失敗時用來分辨：是這部影片的問題（尺寸太大之類），還是顯示卡編碼整個不能用。"""
    try:
        res = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                              "-i", "color=c=black:s=640x360:r=30:d=0.2", *NVENC_ARGS, "-pix_fmt", "yuv420p",
                              "-f", "null", "-"],
                             capture_output=True, timeout=timeout, creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return False
    return res.returncode == 0


def _proxy_args(ffmpeg: str, src: Path, tmp: Path, video_args: list[str]) -> list[str]:
    return [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-map", "0:v:0?", "-map", "0:a:0",
            *video_args, "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(tmp)]


def make_proxy(src: Path, dst: Path, duration: float, on_progress=None, check=None,
               on_note: Callable[[str], None] | None = None):
    """轉成瀏覽器一定能播的 H.264 + AAC。先用顯卡的 NVENC 編碼；NVENC 不能用（或安裝時測過不能用、
    設了 VS_FORCE_NO_NVENC=1）就改用 libx264 -preset veryfast -crf 21，並用 on_note 告訴任務會比較慢。"""
    ffmpeg = config.require_tool("ffmpeg")
    tmp = dst.with_suffix(".tmp.mp4")
    note = X264_NOTE
    try:
        if config.nvenc_allowed() and not _nvenc_failed["value"]:
            try:
                _run_with_progress(_proxy_args(ffmpeg, src, tmp, NVENC_ARGS), duration, on_progress, check)
                tmp.replace(dst)
                return
            except FfmpegError as e:
                if not _nvenc_problem(e.stderr):
                    raise
                safepath.safe_unlink(tmp, dst.parent)
                if check:
                    check()
                if nvenc_works(ffmpeg):
                    # 顯示卡編碼本身能用，只是這部影片不行：這部改用 libx264，之後的照樣先試 NVENC
                    log.warning("NVENC could not encode %s, using libx264 for this file: %s", src.name,
                                e.stderr.strip()[-300:])
                    note = X264_FILE_NOTE
                else:
                    log.warning("NVENC failed, falling back to libx264: %s", e.stderr.strip()[-300:])
                    _nvenc_failed["value"] = True
        if on_note:
            on_note(note)
        if on_progress:
            on_progress(0.0)
        _run_with_progress(_proxy_args(ffmpeg, src, tmp, X264_ARGS), duration, on_progress, check)
    except BaseException:
        safepath.safe_unlink(tmp, dst.parent)  # 取消或失敗時不留半截的暫存檔
        raise
    tmp.replace(dst)
