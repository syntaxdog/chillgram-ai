# services/video_generate.py
from __future__ import annotations

import os
import io
import json
import shutil
import mimetypes
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import requests
from PIL import Image
from google import genai
from google.genai import types
import replicate


# =========================================================
# Path / Storage
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent  # services/ 기준이면 상위로
AI_DIR = BASE_DIR / "ai"


def _ensure_product_dir(product_id: int) -> Path:
    """
    main.py에서 ensure를 갖고 있지만,
    service에서 import하면 순환참조 위험이 있어서 여기서 독립적으로 처리.
    """
    d = BASE_DIR / str(product_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


# =========================================================
# Helpers
# =========================================================
def _download(url: str, out_path: Path) -> Path:
    r = requests.get(url, stream=True, timeout=600)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return out_path


def _ffprobe_duration(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    out = subprocess.check_output(cmd).decode().strip()
    return float(out)


def _concat_with_xfade(video_paths: List[str], out_path: str, fade: float = 0.5) -> str:
    """
    variable length videos -> xfade chain.
    offset for first xfade = dur(v0) - fade
    offset for second xfade = (dur(v0)+dur(v1)) - 2*fade
    ...
    """
    if len(video_paths) == 1:
        shutil.copyfile(video_paths[0], out_path)
        return out_path

    durs = [_ffprobe_duration(v) for v in video_paths]

    cmd = ["ffmpeg", "-y"]
    for v in video_paths:
        cmd += ["-i", v]

    filters = []
    offset = durs[0] - fade
    filters.append(
        f"[0:v][1:v]xfade=transition=fade:duration={fade}:offset={offset}[v01]"
    )

    total = durs[0] + durs[1]
    prev_label = "v01"

    for i in range(2, len(video_paths)):
        offset = total - fade * (i)
        filters.append(
            f"[{prev_label}][{i}:v]xfade=transition=fade:duration={fade}:offset={offset}[v0{i}]"
        )
        prev_label = f"v0{i}"
        total += durs[i]

    filter_complex = ";".join(filters)

    cmd += [
        "-filter_complex",
        filter_complex,
        "-map",
        f"[{prev_label}]",
        "-r",
        "30",
        "-pix_fmt",
        "yuv420p",
        out_path,
    ]

    subprocess.check_call(cmd)
    return out_path


def _veo3_fast_image_to_video(
    image_path: str,
    prompt: str,
    resolution: str = "720p",
    aspect_ratio: str = "9:16",
) -> str:
    with open(image_path, "rb") as f:
        output = replicate.run(
            "google/veo-3-fast",
            input={
                "image": f,
                "prompt": prompt,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
            },
        )

    if isinstance(output, str):
        return output
    if isinstance(output, list) and output:
        return output[0]
    if hasattr(output, "url") and isinstance(output.url, str):
        return output.url

    raise RuntimeError(f"❌ Unexpected veo output: {output}")


def _normalize_scenes_list(plan: Any) -> List[Dict[str, Any]]:
    """
    Gemini 응답이 조금씩 흔들릴 수 있어서 안전하게 scenes list를 추출.
    """
    if isinstance(plan, dict):
        if "scenes" in plan and isinstance(plan["scenes"], list):
            return plan["scenes"]
        if "ad_plan" in plan and isinstance(plan["ad_plan"], list):
            return plan["ad_plan"]
        if all(k in plan for k in ["nano_image_prompt", "video_prompt"]):
            return [plan]
        raise ValueError("Unexpected plan json structure (missing scenes/ad_plan).")
    if isinstance(plan, list):
        return plan
    raise ValueError("Unexpected plan type")


def _get_scene_id(scene: Dict[str, Any], fallback: int) -> int:
    sid = scene.get("scene_number") or scene.get("id") or fallback
    try:
        return int(sid)
    except Exception:
        return fallback


# =========================================================
# Main service function
# =========================================================
async def generate_video_for_product(
    product_id: int,
    req,
    product_image: UploadFile,
) -> Path:
    """
    main.py에서 이렇게 호출하는 전제:

      generate_video_for_product(product_id=..., req=req, product_image=file)

    저장 위치:
      - ai/{product_id}/video/
          - product.png
          - video_plan.json
          - scene_01.png ...
          - scene_01.mp4 ...
          - video.mp4 (최종)
    """

    # -----------------------------------------
    # 0) Env check
    # -----------------------------------------
    GEMINI_KEY = os.environ.get("gemini_api_key", "")
    REPLICATE_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")

    if not GEMINI_KEY:
        raise HTTPException(status_code=500, detail="missing env: gemini_api_key")
    if not REPLICATE_TOKEN:
        raise HTTPException(status_code=500, detail="missing env: REPLICATE_API_TOKEN")

    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_TOKEN
    gclient = genai.Client(api_key=GEMINI_KEY)

    # -----------------------------------------
    # 1) Output dir 결정 (서비스 내부에서)
    # -----------------------------------------
    product_dir = _ensure_product_dir(product_id)
    out_dir = product_dir

    # -----------------------------------------
    # 2) Save uploaded image
    # -----------------------------------------
    img_bytes = await product_image.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="empty product_image")

    filename = product_image.filename or "product.png"
    suffix = Path(filename).suffix.lower()
    if suffix not in [".png", ".jpg", ".jpeg", ".webp"]:
        suffix = ".png"

    product_img_path = out_dir / f"product{suffix}"
    product_img_path.write_bytes(img_bytes)

    ref_mime = mimetypes.guess_type(str(product_img_path))[0] or "image/png"
    ref_bytes = img_bytes

    # -----------------------------------------
    # 3) Gemini: Create plan JSON
    # -----------------------------------------
    SYSTEM = """너는 숏츠 광고 감독이다.
목표: 10초 내외 숏츠 광고를 3개의 씬으로 구성한다.
각 씬은 '제품이 화면 중심에 자연스럽게 등장'해야 하며, 과장된 문구/자막은 금지.
"""

    plan_prompt = f"""
[제품 정보]
- 식품이름: {req.food_name}
- 식품종류: {req.food_type}
- 광고 컨셉: {req.ad_concept}
- 광고 요구사항: {req.ad_req}

[출력]
1개의 씬으로 구성된 광고 기획을 JSON으로만 출력해.
- 전체 길이는 10초 내외가 되도록 duration_hint_sec(각 씬 2~5초 범위 추천)만 제안해.
- 씬은 (도입→클라이맥스→마무리/콜투액션) 흐름으로.
- nano_image_prompt: 참조 이미지(제품 사진)를 기반으로 '같은 제품 패키지/로고/텍스트'가 유지된 채
  인물/배경/연출만 바꾸는 이미지 생성 프롬프트.
- video_prompt: 해당 씬 이미지에서 '짧은 영상'으로 자연스럽게 움직이도록 하는 프롬프트.
"""

    try:
        resp = gclient.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=SYSTEM + "\n\n" + plan_prompt)],
                )
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        plan = json.loads(resp.text)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"gemini plan generation failed: {e}"
        )

    # 저장
    (out_dir / "video_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # scenes normalize
    try:
        scenes_list = _normalize_scenes_list(plan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"invalid plan json: {e}")

    # -----------------------------------------
    # 4) Gemini: Generate scene images
    # -----------------------------------------
    STRICT_IMAGE_RULES = """
[엄격 규칙 - 반드시 준수]
- 제품 패키지, 로고, 원본 글자(텍스트)를 '원본과 동일하게' 유지할 것.
- 글자 왜곡 금지, 가짜 로고 생성 금지, 워터마크 금지, 이미지 내부에 새로운 텍스트 추가 금지.
- 제품 패키지는 같은 디자인/같은 문구/같은 로고로 유지. (새로운 브랜드/철자 생성 금지)
- 배경/인물/조명/카메라 구도만 바꿔서 연출할 것.
""".strip()

    img_paths: List[str] = []
    for idx, sc in enumerate(scenes_list):
        sid = _get_scene_id(sc, fallback=idx + 1)
        nano_prompt = (
            sc["nano_image_prompt"].strip() + "\n\n" + STRICT_IMAGE_RULES
        ).strip()

        try:
            img_resp = gclient.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(text=nano_prompt),
                            types.Part.from_bytes(data=ref_bytes, mime_type=ref_mime),
                        ],
                    )
                ],
                config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"gemini image generation failed (scene {sid}): {e}",
            )

        out_bytes = None
        try:
            for p in img_resp.candidates[0].content.parts:
                if getattr(
                    p, "inline_data", None
                ) and p.inline_data.mime_type.startswith("image/"):
                    out_bytes = p.inline_data.data
                    break
        except Exception:
            out_bytes = None

        if out_bytes is None:
            raise HTTPException(
                status_code=500, detail=f"no image output from gemini (scene {sid})"
            )

        out_path = out_dir / f"scene_{sid:02d}.png"
        Image.open(io.BytesIO(out_bytes)).save(out_path)
        img_paths.append(str(out_path))

    # -----------------------------------------
    # 5) Replicate: Generate scene videos
    # -----------------------------------------
    STRICT_VIDEO_RULES = """
[Strict constraints]
- Keep product package, logo, and original text EXACTLY the same as the reference image.
- NO text deformation, NO fake logo, NO watermark, NO newly generated text.
- Do not add any on-screen text. Captions will be added later.
- Keep the product readable and centered naturally.
""".strip()

    video_paths: List[str] = []
    for idx, sc in enumerate(scenes_list):
        sid = _get_scene_id(sc, fallback=idx + 1)
        prompt_vid = (sc["video_prompt"].strip() + "\n\n" + STRICT_VIDEO_RULES).strip()

        try:
            url = _veo3_fast_image_to_video(
                img_paths[idx],
                prompt_vid,
                resolution="720p",
                aspect_ratio="9:16",
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"replicate veo generation failed (scene {sid}): {e}",
            )

        out_path = out_dir / f"scene_{sid:02d}.mp4"
        try:
            _download(url, out_path)
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"download veo result failed (scene {sid}): {e}",
            )

        video_paths.append(str(out_path))

    # -----------------------------------------
    # 6) FFmpeg: concat with xfade -> final video.mp4
    # -----------------------------------------
    final_mp4 = out_dir / "video.mp4"
    try:
        _concat_with_xfade(video_paths, str(final_mp4), fade=0.5)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500, detail="ffmpeg/ffprobe not installed on server"
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"ffmpeg concat failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"concat failed: {e}")

    return final_mp4
