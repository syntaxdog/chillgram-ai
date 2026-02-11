import asyncio
import json
import os
from pathlib import Path

import aio_pika
from dotenv import load_dotenv
from google.cloud import storage

from services.banner_generate import AdBannerGenerator
from services.sns_image_generate import SNSImageGenerator
from services.video_generate import generate_video_for_product

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "")

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "")
QUEUE_NAME = os.getenv("RABBITMQ_QUEUE", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")

BASE_DIR = Path(__file__).resolve().parent
AI_DIR = BASE_DIR / "ai"


def ensure_product_dir(project_id: int) -> Path:
    d = AI_DIR / str(project_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def job_banner(project_id: int, payload: dict) -> Path:
    product_dir = ensure_product_dir(project_id)
    input_path = product_dir / "package.png"
    output_path = product_dir / "banner.png"

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found (package.png is required)")

    gen = AdBannerGenerator(api_key=API_KEY)
    gen.process(
        image_path=str(input_path),
        headline=payload["headline"],
        typo_text=payload["typo_text"],
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
    out = await generate_video_for_product(project_id=project_id, req=payload, product_image=None)
    return Path(out)


def gcs_upload(local_path: Path, object_name: str) -> str:
    """
    returns gs://bucket/object
    """
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(object_name)

    # content-type 대충 지정(옵션)
    blob.upload_from_filename(str(local_path))
    return f"gs://{GCS_BUCKET}/{object_name}"


async def handle_job(message: aio_pika.IncomingMessage):
    try:
        async with message.process(requeue=False):
            body = message.body.decode("utf-8")
            job = json.loads(body)

            project_id = int(job["projectId"])
            job_type = str(job.get("jobType", "")).upper()
            payload = job.get("payload", {}) or {}

            # 1) 생성
            if job_type == "BANNER":
                out = job_banner(project_id, payload)
                obj = f"{project_id}/banner.png"
            elif job_type == "SNS":
                out = job_sns(project_id, payload)
                obj = f"{project_id}/sns.png"
            elif job_type == "VIDEO":
                out = await job_video(project_id, payload)
                obj = f"{project_id}/video.mp4"
            else:
                raise ValueError(f"unsupported jobType: {job_type}")

            # 2) GCS 업로드
            gs_uri = gcs_upload(out, obj)

            # 3) TODO: 결과 이벤트 발행 / 상태 저장 (Spring이 읽게)
            print(json.dumps({"ok": True, "jobType": job_type, "projectId": project_id, "output": gs_uri}, ensure_ascii=False))
    except Exception as e:
        print(f"[JOB ERROR] {e}", flush=True)
        raise

async def main():
    print(f"Starting RabbitMQ consumer: {RABBITMQ_URL} queue={QUEUE_NAME}")

    connection = await aio_pika.connect_robust(RABBITMQ_URL)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)

    queue = await channel.declare_queue(QUEUE_NAME, durable=True)
    await queue.consume(handle_job)

    # keep running
    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
