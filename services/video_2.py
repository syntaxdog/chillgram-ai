# services/video_2.py
from __future__ import annotations

import os
import io
import json
import time
import asyncio
import shutil
import mimetypes
import subprocess
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import requests
from PIL import Image
from google import genai
from google.genai import types

# Try importing UploadFile from fastapi, fallback to Any if not installed (for local testing without fastapi)
try:
    from fastapi import UploadFile, HTTPException
except ImportError:
    from typing import Any as UploadFile
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

# =========================================================
# Configuration & Constants
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
AI_DIR = BASE_DIR / "ai"

# Logger setup
logger = logging.getLogger(__name__)

# =========================================================
# Helpers from video.py (Ported)
# =========================================================

def _ffprobe_duration(path: str) -> float:
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        out = subprocess.check_output(cmd).decode().strip()
        return float(out)
    except Exception as e:
        logger.error(f"ffprobe duration failed: {e}")
        return 0.0

def _extract_last_frame(video_path: str, out_path: str) -> str:
    """FFmpeg을 사용하여 영상의 마지막 프레임을 이미지로 추출 (video.py 로직)"""
    duration = _ffprobe_duration(video_path)
    # 마지막 지점에서 1.0초 전 프레임을 추출
    seek_time = max(0, duration - 1.0)
    
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(seek_time),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",  # 고화질
        out_path
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path

def upload_image_to_hosting(image_path: str | Path) -> str:
    """이미지를 공개 URL로 업로드 (catbox -> file.io 순서)"""
    image_path = str(image_path)
    if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
        raise RuntimeError(f"이미지 파일 이상: {image_path}")

    with open(image_path, "rb") as f: file_content = f.read()
    
    # 1. catbox.moe
    try:
        files = {"fileToUpload": ("image.png", file_content, "image/png")}
        data = {"reqtype": "fileupload"}
        rb = requests.post("https://catbox.moe/user/api.php", data=data, files=files, timeout=30)
        if rb.status_code == 200:
            return rb.text.strip()
    except Exception:
        pass

    # 2. file.io
    try:
        files = {"file": ("image.png", file_content, "image/png")}
        rb = requests.post("https://file.io/?expires=1d", files=files, timeout=30)
        if rb.status_code == 200:
            data = rb.json()
            if data.get("success"):
                return data.get("link")
    except Exception:
        pass
        
    raise RuntimeError("모든 이미지 호스팅 서비스 업로드 실패")

# =========================================================
# Suno AI Music (Ported from video.py)
# =========================================================

def analyze_image_for_music(gclient, types, img_path: str) -> str:
    """Gemini Vision: 이미지를 분석하여 Suno용 음악 프롬프트 생성"""
    with open(img_path, "rb") as f: img_bytes = f.read()
    
    prompt = """
    Analyze this product image and create a music generation prompt for 'Suno AI' to make a background music for a 30s TV commercial.
    
    [Requirements]
    1. Mood: Matches the product (e.g., Spicy chips -> Exciting/Energetic, Coffee -> Calm/Jazz).
    2. Genre: Commercial BGM, High Quality.
    3. Structure: Starts strong, and **MUST have a natural ending/fade-out EXACTLY at 19 seconds**.
    4. Format: Single line English text description.
    5. Example: "Upbeat pop track, energetic bass, crispy sound textures, spicy mood, 120 BPM, ending at 19s with a cymbal crash."
    """
    
    try:
        resp = gclient.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(role="user", parts=[
                    types.Part(text=prompt),
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                ])
            ]
        )
        return resp.text.strip()
    except Exception:
        return "Upbeat commercial background music, energetic, 30s"

def generate_suno_music(kie_key: str, prompt: str) -> Optional[str]:
    """KIE Suno API를 사용하여 음악 생성"""
    headers = {"Authorization": f"Bearer {kie_key}", "Content-Type": "application/json"}
    payload = {
        "model": "ai-music-api/generate",
        "input": {
            "model": "V4_5PLUS",
            "customMode": True,
            "instrumental": True, 
            "style": prompt[:900], 
            "title": "Ad Background Music",
            "prompt": "", 
            "callBackUrl": "playground"
        }
    }

    try:
        resp = requests.post("https://api.kie.ai/api/v1/jobs/createTask", headers=headers, json=payload, timeout=60)
        if resp.status_code != 200: return None
        task_data = resp.json()
        if task_data.get("code") != 200: return None
             
        task_id = task_data["data"]["id"]
        
        # Polling (Sync inside, but fine for now or could be async)
        for _ in range(60):
            time.sleep(2)
            stat_resp = requests.get(f"https://api.kie.ai/api/v1/jobs/getTask?id={task_id}", headers=headers)
            if stat_resp.status_code == 200:
                stat_data = stat_resp.json()
                status = stat_data["data"]["status"]
                
                if status == "SUCCEEDED":
                    res = stat_data["data"]["response"]
                    try:
                        if isinstance(res, dict):
                            if "audio_urls" in res and res["audio_urls"]: return res["audio_urls"][0]
                            if "audio_clips" in res and res["audio_clips"]:
                                return res["audio_clips"][0].get("audio_url") or res["audio_clips"][0].get("video_url")
                        if isinstance(res, list) and len(res) > 0:
                             if isinstance(res[0], str) and res[0].startswith("http"): return res[0]
                    except Exception:
                        pass
                    return None
                if status == "FAILED": return None
    except Exception:
        return None
    return None

# =========================================================
# Core Logic
# =========================================================

def _ensure_product_dir(product_id: int) -> Path:
    d = BASE_DIR / str(product_id)
    d.mkdir(parents=True, exist_ok=True)
    return d

def _normalize_scenes_list(plan: Any) -> List[Dict[str, Any]]:
    if isinstance(plan, dict):
        if "scenes" in plan and isinstance(plan["scenes"], list): return plan["scenes"]
        if "ad_plan" in plan and isinstance(plan["ad_plan"], list): return plan["ad_plan"]
    if isinstance(plan, list): return plan
    # Fallback structure provided by gemini
    raise ValueError("Invalid plan structure")

async def _generate_video_clip_sora2(kie_key: str, img_path: Path, prompt: str, duration: int = 15) -> str:
    """Sora 2 Generating using URL upload (Ported from video.py)"""
    
    # 1. Upload to Hosting
    image_url = upload_image_to_hosting(img_path)
    if image_url.startswith("http://"): image_url = image_url.replace("http://", "https://")
    
    # Wait for propagation
    await asyncio.sleep(6)
    
    # 2. Prompt Sanitization & Enhancement
    clean_prompt = "".join([c for c in prompt if ord(c) < 128])
    clean_prompt += ", Korean audio, Korean speech, ambience of Korea"
    
    # Sora 2 API Config
    headers = {"Authorization": f"Bearer {kie_key}", "Content-Type": "application/json"}
    payload = {
        "model": "sora-2-image-to-video-stable",
        "input": {
            "image_urls": [image_url],
            "prompt": clean_prompt,
            "duration": str(duration),
            "resolution": "720p",
            "mode": "normal"
        }
    }

    last_error = ""
    # Retry Loop
    for _ in range(3):
        task_id = None
        # Create Task
        for _ in range(3):
            try:
                resp = requests.post("https://api.kie.ai/api/v1/jobs/createTask", headers=headers, json=payload, timeout=60)
                if resp.status_code == 429 or "heavy load" in resp.text.lower():
                    await asyncio.sleep(60); continue
                
                data = resp.json()
                if data.get("code") != 200:
                    if "heavy load" in str(data).lower(): await asyncio.sleep(60); continue
                    await asyncio.sleep(10); continue
                
                task_id = data.get("data", {}).get("taskId")
                if task_id: break
            except Exception:
                await asyncio.sleep(10)
        
        if not task_id: continue

        # Poll Task
        success_url = None
        poll_failed = False
        poll_url = f"https://api.kie.ai/api/v1/jobs/recordInfo?taskId={task_id}"
        
        for _ in range(120): # 10 min
            await asyncio.sleep(5)
            try:
                p_resp = requests.get(poll_url, headers=headers, timeout=30)
                if p_resp.status_code != 200: continue
                p_data = p_resp.json()
                
                state = p_data.get("data", {}).get("state")
                if state == "success":
                    res_json = json.loads(p_data["data"].get("resultJson", "{}"))
                    url_list = res_json.get("resultUrls", [])
                    if url_list:
                        success_url = url_list[0]
                        break
                elif state == "fail":
                    fail_msg = p_data["data"].get("failMsg", "unknown")
                    if "heavy load" in fail_msg.lower():
                        await asyncio.sleep(60); poll_failed = True; break
                    if "safety" in fail_msg.lower(): raise RuntimeError(f"Safety: {fail_msg}")
                    poll_failed = True; break
            except Exception as e:
                if "Safety" in str(e): raise e
                continue
        
        if success_url:
            return success_url # Return URL directly to be downloaded later
        if poll_failed: continue

    raise RuntimeError("Sora 2 Video Generation Failed")

def _concat_and_mix(video_paths: List[str], bgm_path: Optional[str], out_path: str, fade: float = 0.5):
    """FFmpeg Concatenation + BGM Mixing (Ported from video.py)"""
    
    # 1. Check durations & audio
    def get_info(path):
        try:
            cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path]
            dur = float(subprocess.check_output(cmd).decode().strip())
            cmd_a = ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0", path]
            has_audio = bool(subprocess.check_output(cmd_a).decode().strip())
            return dur, has_audio
        except: return 0, False

    infos = [get_info(v) for v in video_paths]
    durs = [i[0] for i in infos]
    
    # Simple Video Crossfade Mapping
    cmd = ["ffmpeg", "-y"]
    for v in video_paths: cmd += ["-i", v]
    
    filter_complex = ""
    # Video Crossfade
    offset = durs[0] - fade
    filter_complex += f"[0:v][1:v]xfade=transition=fade:duration={fade}:offset={offset}[v01];"
    prev_v = "[v01]"
    
    # Audio Mix/Crossfade Logic
    # If BGM exists, mix it. If video has audio, concat it.
    # For simplicity of this port, ensuring basic functionality first:
    # Use the complex filter from video.py specific for 2 clips + BGM
    
    if bgm_path and os.path.exists(bgm_path):
        # Add BGM input
        cmd += ["-i", bgm_path] # Input 2
        bgm_idx = 2
        
        # Audio from clips (assuming they exist or using silent backup is better, but mirroring video.py)
        # Assuming both clips have audio for 'acrossfade'
        # [0:a][1:a]acrossfade...[a_merged]
        # [a_merged][bgm]amix...
        
        filter_complex += f"[0:a][1:a]acrossfade=d={fade}:c1=tri:c2=tri[a_clips];"
        # Trim BGM to 19s and fade out
        filter_complex += f"[{bgm_idx}:a]atrim=0:19,afade=t=out:st=17:d=2,volume=0.3[bgm_trim];"
        filter_complex += f"[a_clips][bgm_trim]amix=inputs=2:duration=first:dropout_transition=2[a_out]"
        
        cmd += [
            "-filter_complex", filter_complex,
            "-map", prev_v,
            "-map", "[a_out]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            out_path
        ]
    else:
        # Video Only or just Video audio
        filter_complex += f"[0:a][1:a]acrossfade=d={fade}:c1=tri:c2=tri[a_out]"
        cmd += [
            "-filter_complex", filter_complex,
            "-map", prev_v,
            "-map", "[a_out]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            out_path
        ]

    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        logger.error(f"Complex mix failed, falling back to simple concat: {e}")
        # Fallback
        with open("concat.txt", "w") as f:
            for v in video_paths: f.write(f"file '{v}'\n")
        subprocess.call(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "concat.txt", "-c", "copy", out_path])


# =========================================================
# Main Entry Point
# =========================================================

async def generate_video_for_product(product_id: int, req: Any, product_image: UploadFile) -> Path:
    """
    Main Service Function (Refactored to use video.py logic)
    - 2 Scenes (30s)
    - Korean Context
    - BGM + Last Frame Continuity
    """
    
    # 0. Env Setup
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    kie_key = os.environ.get("KIE_API_KEY", "")
    if not gemini_key or not kie_key:
        raise HTTPException(status_code=500, detail="Missing API Keys")

    gclient = genai.Client(api_key=gemini_key)
    out_dir = _ensure_product_dir(product_id)
    
    # 1. Save Uploaded Image
    # Handle UploadFile (FastAPI) or bytes
    if hasattr(product_image, "read"):
        img_bytes = await product_image.read()
        filename = getattr(product_image, "filename", "product.png")
    else:
        raise ValueError("Invalid product_image type")
        
    product_img_path = out_dir / f"product-{int(time.time())}.png"
    with open(product_img_path, "wb") as f: f.write(img_bytes)

    # 2. Plan Ad (Gemini) - Using video.py's 2-scene prompt
    SYSTEM_PROMPT = """너는 숙련된 숏츠 광고 감독이다.
목표: 30초 광고, 2개 씬(Clip). [Clip 1: 0~15초] -> [Clip 2: 15~30초]
[필수]
1. Clip 1 (도입): 제품 매력, 맛있게 먹는 모습.
2. Clip 2 (엔딩): 심화 및 엔딩. 마지막 3초는 제품 단독 샷(Hero Shot).
3. 연결성: Clip 1 마지막 장면(인물이 제품을 든 모습)이 Clip 2 시작점이 됨.
4. 인물: 한국인/동양인. 대사는 한국어.
"""
    plan_prompt = f"""
[제품 정보]
이름: {req.food_name} / 종류: {req.food_type}
컨셉: {req.ad_concept} / 요구사항: {req.ad_req}

[출력 형식]
JSON Only.
{{
  "scenes": [
    {{
      "scene_number": 1,
      "duration_hint_sec": 15,
      "scene_script": "Clip 1 한국어 대사",
      "nano_image_prompt": "Clip 1 이미지 프롬프트 (영어)",
      "video_prompt": "Clip 1 영상 프롬프트 (영어)"
    }},
    {{
      "scene_number": 2,
      "duration_hint_sec": 15,
      "scene_script": "Clip 2 한국어 대사",
      "nano_image_prompt": "Clip 2 이미지 프롬프트 (영어, Clip 1 연결성)",
      "video_prompt": "Clip 2 영상 프롬프트 (영어)"
    }}
  ]
}}
"""
    try:
        resp = gclient.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(role="user", parts=[
                    types.Part(text=SYSTEM_PROMPT + "\n" + plan_prompt),
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                ])
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        plan = json.loads(resp.text)
        scenes = _normalize_scenes_list(plan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Planning failed: {e}")

    # 3. Execution Loop
    video_paths = []
    current_ref_img = product_img_path # Path object
    bgm_path = None
    
    for i, sc in enumerate(scenes):
        sid = sc.get("scene_number", i+1)
        
        # A. Image Generation (Gemini)
        prompt = sc["nano_image_prompt"] + "\n\n[STRICT: Product text must match exactly. Korean style.]"
        
        # Prepare content parts
        with open(current_ref_img, "rb") as f: ref_bytes_loop = f.read()
        parts = [
            types.Part(text=prompt),
            types.Part.from_bytes(data=ref_bytes_loop, mime_type="image/png")
        ]
        
        # Generate Image
        scene_img_path = out_dir / f"scene_{sid}.png"
        try:
            img_resp = gclient.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(response_modalities=["IMAGE"])
            )
            # Save Image
            for p in img_resp.candidates[0].content.parts:
                if getattr(p, "inline_data", None):
                    img = Image.open(io.BytesIO(p.inline_data.data))
                    img.save(scene_img_path)
                    break
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image Gen Failed Scene {sid}: {e}")

        # B. Music (Scene 1 only)
        if sid == 1:
            try:
                music_prompt = analyze_image_for_music(gclient, types, str(scene_img_path))
                bgm_url = generate_suno_music(kie_key, music_prompt)
                if bgm_url:
                    bgm_path = out_dir / "bgm.mp3"
                    with open(bgm_path, "wb") as f:
                        f.write(requests.get(bgm_url).content)
            except Exception:
                pass # Ignore music failures

        # C. Video Generation (Sora 2)
        vid_url = await _generate_video_clip_sora2(
            kie_key=kie_key,
            img_path=scene_img_path,
            prompt=sc["video_prompt"],
            duration=15
        )
        
        # Download Video
        vid_path = out_dir / f"scene_{sid}.mp4"
        with open(vid_path, "wb") as f:
            r = requests.get(vid_url, stream=True)
            for chunk in r.iter_content(chunk_size=1024*1024): 
                if chunk: f.write(chunk)
        video_paths.append(str(vid_path))
        
        # D. Extract Last Frame (for next scene)
        if i < len(scenes) - 1:
            last_frame = out_dir / f"scene_{sid}_last.png"
            _extract_last_frame(str(vid_path), str(last_frame))
            current_ref_img = last_frame

    # 4. Final Concat & Mix
    final_mp4 = out_dir / "final_ad.mp4"
    _concat_and_mix(video_paths, str(bgm_path) if bgm_path else None, str(final_mp4))
    
    return final_mp4
