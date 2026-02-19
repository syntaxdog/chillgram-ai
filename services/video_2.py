from __future__ import annotations

import os
import io
import json
import time
import asyncio
import shutil
import logging
import requests
from pathlib import Path
from typing import Any, List, Optional, Dict
from types import SimpleNamespace
from google import genai
from google.genai import types

# FastAPI dependency removed for worker compatibility
from typing import Any as UploadFile

# =========================================================
# Configuration & Constants
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
AI_DIR = BASE_DIR / "ai"
logger = logging.getLogger(__name__)

# [API 키 설정]
KIE_API_KEY = os.environ.get("KIE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# =========================================================
# Async subprocess Helper
# =========================================================
async def run_subprocess(cmd: List[str]) -> bytes:
    """Run subprocess asynchronously with robust encoding handling"""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    
    if proc.returncode != 0:
        try:
            err_msg = stderr.decode("utf-8")
        except UnicodeDecodeError:
            try:
                err_msg = stderr.decode("cp949") 
            except:
                err_msg = stderr.decode("utf-8", errors="replace")

        logger.error(f"Subprocess failed: {err_msg}")
        if "ffmpeg" in cmd[0]: return b""
        raise RuntimeError(f"Command failed: {cmd} \nError: {err_msg}")
        
    return stdout

# =========================================================
# Helpers
# =========================================================
async def _ffprobe_duration(path: str) -> float:
    try:
        # [수정] 오디오/비디오 모두 측정 가능
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path]
        out = await run_subprocess(cmd)
        return float(out.decode().strip())
    except: return 0.0

async def _extract_last_frame(video_path: str, out_path: str) -> str:
    """영상의 마지막 프레임 추출"""
    cmd = ["ffmpeg", "-y", "-sseof", "-3", "-i", video_path, "-update", "1", "-q:v", "2", out_path]
    await run_subprocess(cmd)
    return out_path

async def _create_zoom_outro(image_path: str, duration: float, out_path: str):
    """줌인 아웃트로 영상 생성"""
    duration = max(2.0, duration)
    image_path = image_path.replace("\\", "/")
    
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", image_path,
        "-vf", f"scale=1280:720,zoompan=z='min(zoom+0.0015,1.5)':d={int(duration*30)}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1280x720",
        "-c:v", "libx264",
        "-t", str(duration),
        "-pix_fmt", "yuv420p",
        "-r", "30",
        out_path
    ]
    await run_subprocess(cmd)

async def upload_image_to_hosting(image_path: str | Path) -> str:
    image_path = str(image_path)
    if not os.path.exists(image_path): raise RuntimeError("Image missing")

    def _upload():
        with open(image_path, "rb") as f: content = f.read()
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            resp = requests.post("https://tmpfiles.org/api/v1/upload", files={"file": ("img.png", content, "image/png")}, headers=headers, timeout=30)
            if resp.status_code == 200 and resp.json().get("status") == "success":
                return resp.json()["data"]["url"].replace("http://", "https://").replace("tmpfiles.org/", "tmpfiles.org/dl/")
        except: pass
        try:
            resp = requests.post("https://catbox.moe/user/api.php", data={"reqtype": "fileupload"}, files={"fileToUpload": ("img.png", content, "image/png")}, headers=headers, timeout=30)
            if resp.status_code == 200: return resp.text.strip()
        except: pass
        return None

    url = await asyncio.to_thread(_upload)
    if url: return url
    raise RuntimeError("Image upload failed")

def _ensure_product_dir(product_id: int) -> Path:
    d = AI_DIR / str(product_id)
    d.mkdir(parents=True, exist_ok=True)
    return d

def _normalize_scenes_list(plan: Any) -> List[Dict[str, Any]]:
    if isinstance(plan, dict) and "scenes" in plan: return plan["scenes"]
    return [{"scene_number": 1, "video_prompt": "Product showcase", "duration": 10}]

# =========================================================
# [Gemini] Analyze Visuals & Write Lyrics
# =========================================================
async def analyze_visuals_and_write_lyrics(image_path: Path, product_name: str, concept: str) -> tuple[str, str]:
    default_style = "upbeat k-pop, energetic, bright, female vocals"
    default_lyrics = f"""
    [Verse]
    기분 좋은 바람이 불어오면
    생각나는 그 맛, 정말 특별해
    
    [Chorus]
    언제나 함께해, 나의 최애 간식
    친구들과 나눠 먹는 즐거운 시간
    행복이 가득한 {product_name}!
    사랑해요 {product_name}!
    """

    if not GEMINI_API_KEY: return default_style, default_lyrics

    print(f"👀 Gemini가 영상의 마지막 장면을 분석하고 가사를 씁니다...")
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    with open(image_path, "rb") as f: img_bytes = f.read()

    prompt = f"""
    You are a professional Creative Director.
    Look at this image (the ending of Scene 1). Create a music style and lyrics for a 20-second commercial jingle.
    
    [PRODUCT] {product_name} ({concept})
    
    [LYRICS RULES]
    1. **Length:** Write exactly **6 lines** (Verse 2 lines + Chorus 4 lines). This is the perfect length for a 20-30s song.
    2. **Language:** Catchy Korean lyrics mixed with simple English hooks.
    3. **Structure:**
       - Verse: Brief intro of the vibe.
       - Chorus: Catchy hook with product name.
    4. **CRITICAL:** The very last line MUST end with the product name '{product_name}'.
    
    [OUTPUT JSON]
    {{
      "style_tags": "string (e.g. energetic, k-pop, female vocals)",
      "lyrics": "string (formatted with newlines)"
    }}
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.0-flash", 
                contents=[types.Content(role="user", parts=[types.Part(text=prompt), types.Part.from_bytes(data=img_bytes, mime_type="image/png")])],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            result = json.loads(response.text)
            print(f"🎶 [스타일]: {result['style_tags']}")
            print(f"🎤 [가사]:\n{result['lyrics']}")
            return result['style_tags'], result['lyrics']
            
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = (attempt + 1) * 10
                print(f"⚠️ 사용량 초과(429)! {wait_time}초 대기... ({attempt+1}/{max_retries})")
                await asyncio.sleep(wait_time) # [수정] time.sleep -> await asyncio.sleep
            else:
                print(f"❌ 분석 오류: {e}")
                break

    print("❌ 3번 시도 실패. 비상용 가사를 사용합니다.")
    return default_style, default_lyrics

# =========================================================
# [Suno API] Music Generation
# =========================================================
async def generate_suno_music_with_lyrics(kie_key: str, tags: str, lyrics: str, title: str) -> Optional[str]:
    headers = {"Authorization": f"Bearer {kie_key}", "Content-Type": "application/json"}
    gen_url = "https://api.kie.ai/api/v1/generate"
    info_url = "https://api.kie.ai/api/v1/generate/record-info"
    
    for attempt in range(2):
        model_version = "V3_5" if attempt == 0 else "V3"
        print(f"🚀 Suno 음악 생성 요청... (시도 {attempt+1}/2 - 모델: {model_version})")
        
        def _request():
            try:
                payload = {
                    "model": model_version, 
                    "prompt": lyrics, 
                    "tags": tags, 
                    "title": f"Ad_{title}",
                    "customMode": True, 
                    "instrumental": False, 
                    "callBackUrl": "https://api.kie.ai/playground"
                }
                resp = requests.post(gen_url, headers=headers, json=payload, timeout=30)
                if resp.status_code == 200: return resp.json().get("data", {}).get("taskId")
            except: pass
            return None

        task_id = await asyncio.to_thread(_request)
        if not task_id:
            time.sleep(5); continue
        
        print(f"   ✅ 음악 작업 ID: {task_id}")

        async def _poll():
            for _ in range(60):
                await asyncio.sleep(10)
                try:
                    r = await asyncio.to_thread(requests.get, info_url, headers=headers, params={"taskId": task_id}, timeout=20)
                    if r.status_code == 200:
                        data = r.json().get("data", {})
                        status = data.get("status") or data.get("state")
                        if status in ["SUCCESS", "succeeded", "success"]:
                            res_obj = data.get("response")
                            if isinstance(res_obj, str): res_obj = json.loads(res_obj)
                            suno_data = res_obj.get("sunoData")
                            if suno_data and isinstance(suno_data, list) and len(suno_data) > 0:
                                return suno_data[0].get("audioUrl")
                        elif status in ["FAIL", "failed"]:
                            print(f"   ❌ 생성 실패 (모델 {model_version}): {data.get('failMsg')}")
                            break
                except: pass
            return None

        url = await _poll()
        if url: 
            print(f"   🎯 음악 생성 성공! ({model_version})")
            return url
        time.sleep(5)

    print("❌ 최종 실패: 음악 생성 불가")
    return None

# =========================================================
# Video Generation (Sora 2)
# =========================================================
async def _generate_video_clip_sora2(kie_key: str, image_url: str, prompt: str, duration: int = 10) -> str:
    headers = {"Authorization": f"Bearer {kie_key}", "Content-Type": "application/json"}
    payload = {
        "model": "sora-2-image-to-video-stable",
        "input": {
            "image_urls": [image_url], "prompt": prompt, "duration": str(duration),
            "resolution": "720p", "mode": "normal", "audio": False
        }
    }
    
    print(f"🚀 Sora 영상 생성 요청: {prompt[:30]}...")

    def _generate():
        for attempt in range(3):
            try:
                resp = requests.post("https://api.kie.ai/api/v1/jobs/createTask", headers=headers, json=payload, timeout=60)
                
                # [수정] 실패 원인 로그 출력 강화
                if resp.status_code != 200:
                    print(f"   ⚠️ [시도 {attempt+1}] HTTP 실패 ({resp.status_code}): {resp.text}")
                    time.sleep(5); continue
                
                resp_json = resp.json()
                if resp_json.get("code") != 200:
                     print(f"   ⚠️ [시도 {attempt+1}] API 에러: {resp_json}")
                     time.sleep(5); continue

                task_id = resp_json["data"]["taskId"]
                print(f"   ✅ [시도 {attempt+1}] 작업 ID: {task_id}")

                for _ in range(60):
                    time.sleep(10)
                    r = requests.get(f"https://api.kie.ai/api/v1/jobs/recordInfo?taskId={task_id}", headers=headers, timeout=30)
                    if r.status_code != 200: continue
                    state = r.json()["data"]["state"]
                    if state == "success":
                        result_json_str = r.json()["data"]["resultJson"]
                        if result_json_str and isinstance(result_json_str, str):
                            result_urls = json.loads(result_json_str).get("resultUrls")
                            if result_urls: return result_urls[0]
                    elif state == "fail":
                        fail_msg = r.json()["data"].get("failMsg", "No message")
                        print(f"   ❌ [시도 {attempt+1}] 실패: {fail_msg}")
                        break
            except Exception as e:
                print(f"   ⚠️ [시도 {attempt+1}] 예외 발생: {e}")
                time.sleep(5)
        return None

    url = await asyncio.to_thread(_generate)
    if url: return url
    raise RuntimeError("Video generation failed")

async def _concat_video_list(video_paths: List[str], out_path: str):
    """단순 연결 (재인코딩 적용으로 끊김 방지)"""
    if len(video_paths) < 2:
        return shutil.copy(video_paths[0], out_path)
    
    # [수정] concat_list.txt를 out_path가 있는 폴더에 생성
    out_dir = Path(out_path).parent
    list_file = out_dir / "concat_list.txt"
    
    with open(list_file, "w", encoding='utf-8') as f:
        for v in video_paths:
            safe_path = v.replace("\\", "/")
            f.write(f"file '{safe_path}'\n")
    
    cmd = [
        "ffmpeg", "-y", 
        "-f", "concat", 
        "-safe", "0", 
        "-i", list_file, 
        "-c:v", "libx264", 
        "-preset", "ultrafast", 
        "-pix_fmt", "yuv420p", 
        out_path
    ]
    await run_subprocess(cmd)

async def _mix_audio_video_auto_extend(video_path: str, music_url: Optional[str], image_path: Path, out_path: str):
    """영상+음악 합성 (음악이 길면 연장 + 페이드아웃 자동 계산)"""
    # [수정] 임시 파일들을 out_path가 있는 폴더(product_dir) 내부에 생성하여 권한 문제 방지
    out_dir = Path(out_path).parent
    bgm_path = out_dir / "bgm_temp.mp3"
    
    if not music_url:
        return shutil.copy(video_path, out_path)

    try:
        content = await asyncio.to_thread(requests.get, music_url)
        with open(bgm_path, "wb") as f: f.write(content.content)
    except:
        return shutil.copy(video_path, out_path)

    vid_dur = await _ffprobe_duration(video_path)
    mus_dur = await _ffprobe_duration(str(bgm_path))
    
    print(f"⏱ 길이 비교: 영상 {vid_dur}초 vs 음악 {mus_dur}초")
    
    final_video_input = video_path
    final_duration = vid_dur
    
    # 임시 파일 경로들
    outro_path = out_dir / "outro_zoom.mp4"
    merged_path = out_dir / "merged_visual.mp4"

    if mus_dur > vid_dur + 0.1:
        gap = mus_dur - vid_dur
        print(f"🎬 음악이 {gap:.1f}초 더 깁니다. 줌인 아웃트로 생성...")
        
        await _create_zoom_outro(str(image_path), gap, str(outro_path))
        await _concat_video_list([video_path, str(outro_path)], str(merged_path))
        
        if merged_path.exists():
            merged_dur = await _ffprobe_duration(str(merged_path))
            if merged_dur > vid_dur: 
                final_video_input = str(merged_path)
                final_duration = merged_dur
    else:
        final_duration = min(vid_dur, mus_dur)

    fade_start = max(0, final_duration - 2)
    print(f"🎚️ 오디오 페이드아웃 시작: {fade_start}초")

    cmd = [
        "ffmpeg", "-y",
        "-i", final_video_input,
        "-i", str(bgm_path),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy",
        "-af", f"afade=t=out:st={fade_start}:d=2", 
        "-shortest",
        out_path
    ]
    
    try:
        await run_subprocess(cmd)
    except Exception:
        shutil.copy(video_path, out_path)
    
    # [수정] 임시 파일 정리 (Path 객체 사용)
    for f in [bgm_path, outro_path, merged_path]:
        if f.exists(): 
            try: os.remove(f)
            except: pass
    
    # concat_list.txt는 _concat_video_list 내부에서 처리되므로 여기선 생략하거나 별도 처리
    concat_list = out_dir / "concat_list.txt"
    if concat_list.exists():
        try: os.remove(concat_list)
        except: pass

# =========================================================
# Main Entry Point
# =========================================================
async def generate_video_for_product(product_id: int, req: Any, product_image: UploadFile) -> Path:
    if not GEMINI_API_KEY or not KIE_API_KEY: raise ValueError("Keys missing")

    gclient = genai.Client(api_key=GEMINI_API_KEY)
    out_dir = _ensure_product_dir(product_id)
    if isinstance(req, dict): req = SimpleNamespace(**req)
    
    if isinstance(product_image, bytes): img_bytes = product_image
    elif hasattr(product_image, "read"): img_bytes = await product_image.read()
    else: raise ValueError("Invalid image")
    
    product_img_path = out_dir / f"product_origin.png"
    with open(product_img_path, "wb") as f: f.write(img_bytes)

    # 1. 정보 추론 (누락 시) - [수정] 여기서 async 대기 사용
    if not getattr(req, "food_name", None) or not getattr(req, "ad_concept", None):
        print("⚠️ 제품 정보 누락! Gemini가 이미지를 보고 정보를 추론합니다...")
        
        infer_prompt = """
        Analyze this product image and extract info for a commercial.
        Output JSON:
        {
            "food_name": "Product Name (Korean)",
            "food_type": "Category (e.g. Snack, Drink)",
            "ad_concept": "Best matching ad concept (e.g. Refreshing, Spicy, Premium)",
            "ad_req": "Key selling points"
        }
        """
        for attempt in range(4):
            try:
                resp_inf = await asyncio.to_thread(
                    gclient.models.generate_content,
                    model="gemini-2.0-flash", 
                    contents=[types.Content(role="user", parts=[types.Part(text=infer_prompt), types.Part.from_bytes(data=img_bytes, mime_type="image/png")])],
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                inferred = json.loads(resp_inf.text)
                
                if not getattr(req, "food_name", None): req.food_name = inferred.get("food_name", "Unknown Product")
                if not getattr(req, "food_type", None): req.food_type = inferred.get("food_type", "Food")
                if not getattr(req, "ad_concept", None): req.ad_concept = inferred.get("ad_concept", "Delicious")
                if not getattr(req, "ad_req", None): req.ad_req = inferred.get("ad_req", "Showcase the product")
                
                print(f"✅ 추론 완료: {req.food_name} / {req.ad_concept}")
                break 
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < 3:
                        wait_time = (attempt + 1) * 10 
                        print(f"⚠️ 사용량 초과(429)! {wait_time}초 쉬고 추론 재시도... ({attempt+1}/4)")
                        await asyncio.sleep(wait_time) # [핵심] 여기서 await로 대기해야 타임아웃 오류 안 남
                        continue
                
                if attempt == 3 or ("429" not in str(e) and "RESOURCE_EXHAUSTED" not in str(e)):
                    print(f"❌ 추론 실패, 기본값 사용: {e}")
                    if not getattr(req, "food_name", None): req.food_name = "맛있는 제품"
                    if not getattr(req, "food_type", None): req.food_type = "음식"
                    if not getattr(req, "ad_concept", None): req.ad_concept = "맛있게 먹는 모습"
                    if not getattr(req, "ad_req", None): req.ad_req = "제품 강조"
                    break

    SYSTEM_PROMPT = """너는 숙련된 숏츠 광고 감독이다.
목표: 20초 광고, 2개 씬(Clip). [Clip 1: 0~10초] -> [Clip 2: 10~20초]
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
      "duration_hint_sec": 10,
      "video_prompt": "Clip 1 영상 프롬프트 (영어)"
    }},
    {{
      "scene_number": 2,
      "duration_hint_sec": 10,
      "video_prompt": "Clip 2 영상 프롬프트 (영어)"
    }}
  ]
}}
"""
    
    async def _plan():
        resp = gclient.models.generate_content(
            model="gemini-2.0-flash",
            contents=[types.Content(role="user", parts=[types.Part(text=SYSTEM_PROMPT + "\n" + plan_prompt), types.Part.from_bytes(data=img_bytes, mime_type="image/png")])],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(resp.text)
    
    try:
        plan = await _plan()
        scenes = _normalize_scenes_list(plan)
    except:
        scenes = [{"scene_number": 1, "video_prompt": "Intro", "duration": 10}, {"scene_number": 2, "video_prompt": "Climax", "duration": 10}]

    video_paths = []
    
    # 3. [STEP 1] Scene 1 생성
    print("🚀 [Step 1] Scene 1 생성 중...")
    sc1 = scenes[0]
    img_url_1 = await upload_image_to_hosting(product_img_path)
    vid_url_1 = await _generate_video_clip_sora2(KIE_API_KEY, img_url_1, sc1["video_prompt"] + " [Korean model]", 10)
    
    vid_path_1 = out_dir / "scene_1.mp4"
    with open(vid_path_1, "wb") as f: f.write(requests.get(vid_url_1).content)
    video_paths.append(str(vid_path_1))

    # 4. [STEP 2] 분석 & Scene 2
    print("🚀 [Step 2] Scene 1 종료 장면 분석...")
    last_frame_path = out_dir / "scene_1_last.png"
    await _extract_last_frame(str(vid_path_1), str(last_frame_path))
    
    music_style, music_lyrics = await analyze_visuals_and_write_lyrics(last_frame_path, req.food_name, req.ad_concept)

    print("🚀 [Step 3] Scene 2와 음악 병렬 생성...")
    async def task_scene_2():
        img_url_2 = await upload_image_to_hosting(last_frame_path)
        sc2 = scenes[1] if len(scenes) > 1 else scenes[0]
        ending_prompt = " [Camera: Slow linger, smooth fade out feeling] [Emotion: Satisfaction]"
        v_url = await _generate_video_clip_sora2(KIE_API_KEY, img_url_2, sc2["video_prompt"] + ending_prompt, 10)
        p = out_dir / "scene_2.mp4"
        with open(p, "wb") as f: f.write(requests.get(v_url).content)
        return str(p)

    async def task_music():
        return await generate_suno_music_with_lyrics(KIE_API_KEY, music_style, music_lyrics, req.food_name)

    results = await asyncio.gather(task_scene_2(), task_music())
    vid_path_2, music_url = results[0], results[1]
    video_paths.append(vid_path_2)

    # 6. [STEP 4] 최종 합체
    print("🚀 [Step 4] 최종 편집 (스마트 아웃트로)...")
    stitched_mp4 = out_dir / "stitched_raw.mp4"
    await _concat_video_list(video_paths, str(stitched_mp4))
    
    final_mp4 = out_dir / "final_ad_video.mp4"
    await _mix_audio_video_auto_extend(str(stitched_mp4), music_url, product_img_path, str(final_mp4))
    
    print(f"🎉 모든 작업 완료! 최종 파일: {final_mp4}")
    return final_mp4