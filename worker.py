import argparse
import json
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

# 기존 services 재사용
from services.banner_generate import AdBannerGenerator
from services.sns_image_generate import SNSImageGenerator
from services.video_generate import generate_video_for_product

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "")

BASE_DIR = Path(__file__).resolve().parent
AI_DIR = BASE_DIR / "ai"


def ensure_product_dir(project_id: int) -> Path:
    product_dir = AI_DIR / str(project_id)
    product_dir.mkdir(parents=True, exist_ok=True)
    return product_dir


def job_banner(project_id: int, payload: dict) -> Path:
    product_dir = ensure_product_dir(project_id)
    input_path = product_dir / "package.png"
    output_path = product_dir / "banner.png"

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found (package.png is required)")

    headline = payload["headline"]
    typo_text = payload["typo_text"]

    gen = AdBannerGenerator(api_key=API_KEY)
    gen.process(
        image_path=str(input_path),
        headline=headline,
        typo_text=typo_text,
        output_path=str(output_path),
    )
    return output_path


def job_sns(project_id: int, payload: dict) -> Path:
    product_dir = ensure_product_dir(project_id)
    product_path = product_dir / "package.png"
    if not product_path.exists():
        raise FileNotFoundError("package.png not found. Upload/generate package first.")

    background_path = product_dir / "sns_background.png"
    final_path = product_dir / "sns.png"

    gen = SNSImageGenerator()
    gen.generate(
        product_path=str(product_path),
        main_text=payload["main_text"],
        sub_text=payload.get("sub_text", "") or "",
        preset=payload.get("preset"),
        custom_prompt=payload.get("custom_prompt"),
        guideline=payload.get("guideline"),
        output_path=str(final_path),
        save_background=payload.get("save_background", True),
        background_output_path=str(background_path) if payload.get("save_background", True) else None,
    )
    return final_path


async def job_video(project_id: int, payload: dict) -> Path:
    """
    현재 네 video_generate.generate_video_for_product가 UploadFile 기반이면,
    여기서 바로 호출이 깨질 가능성이 높다.
    우선은 'package.png가 이미 존재한다' 전제로 video_generate 쪽을 path 기반으로 바꾸는 게 정석.
    """
    product_dir = ensure_product_dir(project_id)

    # video 생성에 필요한 메타
    # payload 예:
    # { "food_name": "...", "food_type": "...", "ad_concept": "...", "ad_req": "..." }
    # generate_video_for_product가 pydantic 모델을 원하면 내부에서 변환해야 함.
    final_mp4_path = await generate_video_for_product(project_id=project_id, req=payload, product_image=None)
    # 위 함수가 Path/string 반환한다고 가정
    return Path(final_mp4_path)


def preload_package_png(project_id: int, package_src: str | None):
    """
    컨테이너 테스트를 위해 package.png가 필요하면 host 파일을 볼륨으로 마운트해서 복사.
    """
    if not package_src:
        return
    src = Path(package_src)
    if not src.exists():
        raise FileNotFoundError(f"package_src not found: {src}")
    product_dir = ensure_product_dir(project_id)
    dst = product_dir / "package.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def load_job(job_json_path: str) -> dict:
    p = Path(job_json_path)
    if not p.exists():
        raise FileNotFoundError(f"job json not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, help="path to job json file")
    parser.add_argument("--package-src", required=False, help="optional path to package.png to seed ai/{projectId}/package.png")
    args = parser.parse_args()

    job = load_job(args.job)
    project_id = int(job["projectId"])
    job_type = job["jobType"]  # banner|sns|video
    payload = job.get("payload", {})

    # 테스트 편의: package.png를 미리 넣을 수 있게
    preload_package_png(project_id, args.package_src)

    if job_type == "banner":
        out = job_banner(project_id, payload)
        print(json.dumps({"ok": True, "type": "banner", "output": str(out)}, ensure_ascii=False))
        return

    if job_type == "sns":
        out = job_sns(project_id, payload)
        print(json.dumps({"ok": True, "type": "sns", "output": str(out)}, ensure_ascii=False))
        return

    if job_type == "video":
        import asyncio
        out = asyncio.run(job_video(project_id, payload))
        print(json.dumps({"ok": True, "type": "video", "output": str(out)}, ensure_ascii=False))
        return

    raise ValueError(f"unsupported jobType: {job_type}")


if __name__ == "__main__":
    main()