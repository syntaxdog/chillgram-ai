import asyncio
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

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


def pick(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return first existing non-None key among candidates."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def parse_job(body: str) -> Dict[str, Any]:
    """
    Supports:
    1) Plain job:
       {"projectId":1,"jobType":"BANNER","payload":{...}}
    2) Debezium outbox envelope:
       {"schema":{...},"payload":"{...job json...}"}
       or payload as dict (rare)
    """
    outer = json.loads(body)

    if isinstance(outer, dict) and "payload" in outer and "schema" in outer:
        inner = outer.get("payload")
        # Debezium payload is often a JSON string
        if isinstance(inner, str):
            return json.loads(inner)
        if isinstance(inner, dict):
            return inner
        raise ValueError(f"Unsupported Debezium payload type: {type(inner)}")

    if not isinstance(outer, dict):
        raise ValueError(f"Job must be an object, got {type(outer)}")

    return outer


def normalize_payload(job_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept both snake_case and camelCase payload fields.
    Return a normalized snake_case payload for internal job_* functions.
    """
    if job_type == "BANNER":
        headline = pick(payload, "headline")
        typo_text = pick(payload, "typo_text", "typoText")
        if not headline or not typo_text:
            raise ValueError(f"BANNER payload missing fields: headline={headline}, typo_text={typo_text}")
        return {"headline": headline, "typo_text": typo_text}

    if job_type == "SNS":
        main_text = pick(payload, "main_text", "mainText")
        sub_text = pick(payload, "sub_text", "subText", default="")
        preset = pick(payload, "preset")
        custom_prompt = pick(payload, "custom_prompt", "customPrompt")
        guideline = pick(payload, "guideline")
        save_background = pick(payload, "save_background", "saveBackground", default=True)

        if not main_text:
            raise ValueError("SNS payload missing field: main_text/mainText")

        return {
            "main_text": main_text,
            "sub_text": sub_text or "",
            "preset": preset,
            "custom_prompt": custom_prompt,
            "guideline": guideline,
            "save_background": bool(save_background),
        }

    if job_type == "VIDEO":
        # 영상은 지금 너 코드상 payload 그대로 req로 넘기고 있으니 일단 패스(원하면 여기도 정규화 가능)
        return payload

    return payload


def job_banner(project_id: int, payload: Dict[str, Any]) -> Path:
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


def job_sns(project_id: int, payload: Dict[str, Any]) -> Path:
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


async def job_video(project_id: int, payload: Dict[str, Any]) -> Path:
    out = await generate_video_for_product(project_id=project_id, req=payload, product_image=None)
    return Path(out)


def gcs_upload(local_path: Path, object_name: str) -> str:
    """
    returns gs://bucket/object
    """
    if not GCS_BUCKET:
        raise ValueError("GCS_BUCKET is empty")

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(str(local_path))
    return f"gs://{GCS_BUCKET}/{object_name}"


async def handle_job(message: aio_pika.IncomingMessage):
    raw = message.body.decode("utf-8", errors="replace")
    print(f"[RECEIVED] {raw}", flush=True)

    async with message.process(requeue=False):
        try:
            job = parse_job(raw)

            # projectId는 Debezium inner payload 안에 있음(파싱 후 job에서 읽힘)
            project_id = int(job["projectId"])
            job_type = str(job.get("jobType", "")).upper()
            payload = job.get("payload") or {}

            if not job_type:
                raise ValueError("jobType is empty")
            if not isinstance(payload, dict):
                raise ValueError(f"payload must be object, got {type(payload)}")

            payload_norm = normalize_payload(job_type, payload)

            # 1) 생성
            if job_type == "BANNER":
                out = job_banner(project_id, payload_norm)
                obj = f"{project_id}/banner.png"
            elif job_type == "SNS":
                out = job_sns(project_id, payload_norm)
                obj = f"{project_id}/sns.png"
            elif job_type == "VIDEO":
                out = await job_video(project_id, payload_norm)
                obj = f"{project_id}/video.mp4"
            else:
                raise ValueError(f"unsupported jobType: {job_type}")

            # 2) GCS 업로드
            gs_uri = gcs_upload(out, obj)

            # 3) 로그(최소)
            print(
                json.dumps(
                    {"ok": True, "jobType": job_type, "projectId": project_id, "output": gs_uri},
                    ensure_ascii=False,
                ),
                flush=True,
            )

        except Exception:
            print("[JOB ERROR] job failed", flush=True)
            traceback.print_exc()
            # raise 해야 message.process가 reject 처리(현재 requeue=False라 DLQ 없으면 드랍)
            raise


async def main():
    if not RABBITMQ_URL:
        raise ValueError("RABBITMQ_URL is empty")
    if not QUEUE_NAME:
        raise ValueError("RABBITMQ_QUEUE is empty")

    print(f"Starting RabbitMQ consumer: {RABBITMQ_URL} queue={QUEUE_NAME}", flush=True)

    connection = await aio_pika.connect_robust(RABBITMQ_URL)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)

    queue = await channel.declare_queue(QUEUE_NAME, durable=True)
    await queue.consume(handle_job)

    # keep running
    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
