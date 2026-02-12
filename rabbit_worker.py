import asyncio
import json
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import aio_pika
from aio_pika import DeliveryMode, Message, IncomingMessage
from dotenv import load_dotenv
from google.cloud import storage

from services.banner_generate import AdBannerGenerator
from services.sns_image_generate import SNSImageGenerator
from services.video_generate import generate_video_for_product

load_dotenv()


@dataclass(frozen=True)
class Env:
    api_key: str
    rabbitmq_url: str
    queue_jobs: str
    queue_results: str
    gcs_bucket: str


def load_env() -> Env:
    return Env(
        api_key=os.getenv("GEMINI_API_KEY", ""),
        rabbitmq_url=os.getenv("RABBITMQ_URL", ""),
        queue_jobs=os.getenv("RABBITMQ_QUEUE", ""),
        queue_results=os.getenv("RABBITMQ_RESULT_QUEUE", "chillgram.job-results"),
        gcs_bucket=os.getenv("GCS_BUCKET", ""),
    )


BASE_DIR = Path(__file__).resolve().parent
AI_DIR = BASE_DIR / "ai"


def ensure_project_dir(project_id: int) -> Path:
    d = AI_DIR / str(project_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def pick(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def parse_job_message(raw: str) -> Dict[str, Any]:
    """
    Supports:
    - Plain job JSON (most likely with Debezium outbox router)
    - Debezium envelope:
      {"schema":{...},"payload":"{...job json...}"}
    """
    outer = json.loads(raw)

    # Debezium envelope
    if isinstance(outer, dict) and "schema" in outer and "payload" in outer:
        inner = outer["payload"]
        if isinstance(inner, str):
            return json.loads(inner)
        if isinstance(inner, dict):
            return inner
        raise ValueError(f"Unsupported Debezium payload type: {type(inner)}")

    if not isinstance(outer, dict):
        raise ValueError(f"Job must be an object, got {type(outer)}")

    return outer


def normalize_payload(job_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    job_type = job_type.upper()

    if job_type == "BANNER":
        headline = pick(payload, "headline")
        typo_text = pick(payload, "typo_text", "typoText")
        if not headline or not typo_text:
            raise ValueError(f"BANNER payload missing: headline={headline}, typoText={typo_text}")
        return {"headline": headline, "typo_text": typo_text}

    if job_type == "SNS":
        main_text = pick(payload, "main_text", "mainText")
        sub_text = pick(payload, "sub_text", "subText", default="")
        preset = pick(payload, "preset")
        custom_prompt = pick(payload, "custom_prompt", "customPrompt")
        guideline = pick(payload, "guideline")
        save_background = pick(payload, "save_background", "saveBackground", default=True)

        if not main_text:
            raise ValueError("SNS payload missing: mainText/main_text")

        return {
            "main_text": main_text,
            "sub_text": sub_text or "",
            "preset": preset,
            "custom_prompt": custom_prompt,
            "guideline": guideline,
            "save_background": bool(save_background),
        }

    if job_type == "VIDEO":
        return payload

    return payload


class GcsUploader:
    def __init__(self, bucket: str):
        if not bucket:
            raise ValueError("GCS_BUCKET is empty")
        self.bucket_name = bucket
        self.client = storage.Client()

    def upload(self, local_path: Path, object_name: str) -> str:
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(object_name)
        blob.upload_from_filename(str(local_path))
        return f"gs://{self.bucket_name}/{object_name}"


class JobRunner:
    def __init__(self, api_key: str, uploader: GcsUploader):
        self.api_key = api_key
        self.uploader = uploader

    def run_banner(self, project_id: int, payload: Dict[str, Any]) -> Tuple[Path, str]:
        d = ensure_project_dir(project_id)
        input_path = d / "package.png"
        output_path = d / "banner.png"

        if not input_path.exists():
            raise FileNotFoundError(f"{input_path} not found (package.png required)")

        gen = AdBannerGenerator(api_key=self.api_key)
        gen.process(
            image_path=str(input_path),
            headline=payload["headline"],
            typo_text=payload["typo_text"],
            output_path=str(output_path),
        )
        return output_path, f"{project_id}/banner.png"

    def run_sns(self, project_id: int, payload: Dict[str, Any]) -> Tuple[Path, str]:
        d = ensure_project_dir(project_id)
        product_path = d / "package.png"
        if not product_path.exists():
            raise FileNotFoundError("package.png not found. Upload/generate package first.")

        background_path = d / "sns_background.png"
        final_path = d / "sns.png"

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
        return final_path, f"{project_id}/sns.png"

    async def run_video(self, project_id: int, payload: Dict[str, Any]) -> Tuple[Path, str]:
        out = await generate_video_for_product(project_id=project_id, req=payload, product_image=None)
        return Path(out), f"{project_id}/video.mp4"

    async def execute(self, job: Dict[str, Any]) -> str:
        job_type = str(job.get("jobType", "")).upper()
        project_id = int(job["projectId"])
        payload = job.get("payload") or {}
        if not isinstance(payload, dict):
            raise ValueError(f"payload must be object, got {type(payload)}")

        payload_norm = normalize_payload(job_type, payload)

        if job_type == "BANNER":
            local_path, obj = self.run_banner(project_id, payload_norm)
        elif job_type == "SNS":
            local_path, obj = self.run_sns(project_id, payload_norm)
        elif job_type == "VIDEO":
            local_path, obj = await self.run_video(project_id, payload_norm)
        else:
            raise ValueError(f"unsupported jobType: {job_type}")

        return self.uploader.upload(local_path, obj)


class ResultPublisher:
    def __init__(self, channel: aio_pika.Channel, result_queue: str):
        self.channel = channel
        self.result_queue = result_queue

    async def publish(self, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        msg = Message(
            body=body,
            content_type="application/json",
            delivery_mode=DeliveryMode.PERSISTENT,
        )
        await self.channel.default_exchange.publish(msg, routing_key=self.result_queue)


async def handle_message(msg: IncomingMessage, runner: JobRunner, pub: ResultPublisher):
    raw = msg.body.decode("utf-8", errors="replace")
    print(f"[RECEIVED] {raw}", flush=True)

    async with msg.process(requeue=False):
        job_id = ""
        try:
            job = parse_job_message(raw)

            job_id = str(job.get("jobId") or "").strip()
            if not job_id:
                raise ValueError("jobId is empty")

            # 실행
            gs_uri = await runner.execute(job)

            # 성공 결과
            await pub.publish({
                "jobId": job_id,
                "success": True,
                "outputUri": gs_uri,
                "errorCode": None,
                "errorMessage": None,
            })

            print(json.dumps({"ok": True, "jobId": job_id, "output": gs_uri}, ensure_ascii=False), flush=True)

        except Exception as e:
            print("[JOB ERROR] job failed", flush=True)
            traceback.print_exc()

            # 실패 결과(best effort)
            try:
                await pub.publish({
                    "jobId": job_id,
                    "success": False,
                    "outputUri": None,
                    "errorCode": "WORKER_FAILED",
                    "errorMessage": str(e) if e else "unknown error",
                })
            except Exception:
                print("[RESULT PUBLISH ERROR] failed to publish job result", flush=True)
                traceback.print_exc()


async def main():
    env = load_env()
    if not env.rabbitmq_url:
        raise ValueError("RABBITMQ_URL is empty")
    if not env.queue_jobs:
        raise ValueError("RABBITMQ_QUEUE is empty")

    print(
        f"Starting consumer: url={env.rabbitmq_url} jobs={env.queue_jobs} results={env.queue_results}",
        flush=True,
    )

    connection = await aio_pika.connect_robust(env.rabbitmq_url)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)

    # queues
    await channel.declare_queue(env.queue_jobs, durable=True)
    await channel.declare_queue(env.queue_results, durable=True)

    uploader = GcsUploader(env.gcs_bucket)
    runner = JobRunner(env.api_key, uploader)
    publisher = ResultPublisher(channel, env.queue_results)

    q = await channel.get_queue(env.queue_jobs)
    await q.consume(lambda m: handle_message(m, runner, publisher))

    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
