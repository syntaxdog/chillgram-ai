import asyncio
import json
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import aio_pika
from aio_pika import DeliveryMode, Message, IncomingMessage
from dotenv import load_dotenv
from google.cloud import storage

from services.banner_row import AdBannerGenerator
from services.sns_image_generate import SNSImageGenerator
from services.video_2 import generate_video_for_product
from services.dieline_generate import DielineGenerator
from services.package_generate import PackageGenerator  # ✅ 추가

load_dotenv()

# =========================
# Env
# =========================
@dataclass(frozen=True)
class Env:
    api_key: str
    rabbitmq_url: str
    queue_jobs: str
    queue_results: str
    gcs_bucket: str
    video_timeout_sec: int
    gcs_public_base_url: str  # 예: https://storage.googleapis.com/<bucket>


def load_env() -> Env:
    bucket = os.getenv("GCS_BUCKET", "").strip()
    public_base = os.getenv("GCS_PUBLIC_BASE_URL", "").strip()
    if not public_base and bucket:
        public_base = f"https://storage.googleapis.com/{bucket}"
    return Env(
        api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        rabbitmq_url=os.getenv("RABBITMQ_URL", "").strip(),
        queue_jobs=os.getenv("RABBITMQ_QUEUE", "").strip(),
        queue_results=os.getenv("RABBITMQ_RESULT_QUEUE", "chillgram.job-results").strip(),
        gcs_bucket=bucket,
        video_timeout_sec=int(os.getenv("VIDEO_TIMEOUT_SEC", "900")),
        gcs_public_base_url=public_base,
    )


BASE_DIR = Path(__file__).resolve().parent
AI_DIR = BASE_DIR / "ai"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_project_dir(project_id: int) -> Path:
    return ensure_dir(AI_DIR / str(project_id))


def pick(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def parse_job_message(raw: str) -> Dict[str, Any]:
    """
    Supports:
    - Plain job JSON
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


# =========================
# Payload normalize
# =========================
def normalize_payload(job_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    jt = job_type.upper()

    if jt == "BANNER":
        headline = pick(payload, "headline")
        typo_text = pick(payload, "typo_text", "typoText")
        ratio = pick(payload, "ratio", "Ratio")
        guideline = pick(payload, "guideline", "Guideline", default="")
        if not headline or not typo_text:
            raise ValueError(f"BANNER payload 필수값 누락: headline={headline}, typoText={typo_text}")
        if not ratio:
            raise ValueError("BANNER payload 필수값 누락: ratio")
        return {"headline": headline, "typo_text": typo_text, "ratio": ratio, "guideline": guideline}

    if jt == "SNS":
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

    if jt == "VIDEO":
        food_name = pick(payload, "food_name", "foodName")
        food_type = pick(payload, "food_type", "foodType")
        ad_concept = pick(payload, "ad_concept", "adConcept")
        ad_req = pick(payload, "ad_req", "adReq")
        if not (food_name and food_type and ad_concept and ad_req):
            raise ValueError(
                "VIDEO payload must match VideoGenRequest: "
                "{food_name/foodName, food_type/foodType, ad_concept/adConcept, ad_req/adReq}"
            )
        return {"food_name": food_name, "food_type": food_type, "ad_concept": ad_concept, "ad_req": ad_req}

    if jt == "DIELINE":
        prompt = pick(payload, "prompt", "instruction", default="")
        concept_file = pick(payload, "concept_file", "conceptFile", default="package_input.png")
        return {"prompt": prompt, "concept_file": concept_file}

    # ✅ BASIC = (기존 PACKAGE) 패키지 생성/편집
    # Spring payload: { prompt: "..."} or { instruction: "..." }
    if jt == "BASIC":
        prompt = pick(payload, "prompt", "instruction")
        if not prompt:
            raise ValueError("BASIC payload missing: prompt/instruction")
        return {"prompt": prompt}

    return payload


# =========================
# GCS helpers
# =========================
def parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
    # gs://bucket/path/to.obj
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"not gs uri: {gs_uri}")
    no_scheme = gs_uri[len("gs://") :]
    slash = no_scheme.find("/")
    if slash < 0:
        raise ValueError(f"invalid gs uri: {gs_uri}")
    bucket = no_scheme[:slash]
    obj = no_scheme[slash + 1 :]
    if not obj:
        raise ValueError(f"invalid gs uri object empty: {gs_uri}")
    return bucket, obj


class GcsUploader:
    def __init__(self, bucket: str, public_base_url: str):
        if not bucket:
            raise ValueError("GCS_BUCKET is empty")
        if not public_base_url:
            raise ValueError("GCS_PUBLIC_BASE_URL is empty")
        self.bucket_name = bucket
        self.public_base_url = public_base_url.rstrip("/")
        self.client = storage.Client()

    def upload_file(self, local_path: Path, object_name: str, content_type: Optional[str] = None) -> Tuple[str, str]:
        """returns (gs_uri, public_url)"""
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(object_name)
        if content_type:
            blob.content_type = content_type
        blob.upload_from_filename(str(local_path))
        gs_uri = f"gs://{self.bucket_name}/{object_name}"
        public_url = f"{self.public_base_url}/{object_name}"
        return gs_uri, public_url


# =========================
# Runner
# =========================
class JobRunner:
    def __init__(self, api_key: str, uploader: GcsUploader, video_timeout_sec: int):
        self.api_key = api_key
        self.uploader = uploader
        self.video_timeout_sec = video_timeout_sec

    def run_banner(self, project_id: int, payload: Dict[str, Any]) -> Tuple[Path, str, str]:
        d = ensure_project_dir(project_id)
        input_path = d / "package.png"
        
        if not input_path.exists():
            raise FileNotFoundError(f"{input_path} not found (package.png required)")

        gen = AdBannerGenerator(api_key=self.api_key, ratio=payload["ratio"])
        
        # Use fixed filename "banner.png" as requested
        filename = "banner.png"
        output_path = d / filename

        gen.process(
            image_path=str(input_path),
            output_path=str(output_path),
            typo_text=payload["typo_text"],
            guideline=payload["guideline"] or payload["headline"]
        )
        return output_path, f"{project_id}/{filename}", "image/png"

    def run_sns(self, project_id: int, payload: Dict[str, Any]) -> Tuple[Path, str, str]:
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
        return final_path, f"{project_id}/sns.png", "image/png"

    def run_dieline(self, project_id: int, payload: Dict[str, Any]) -> Tuple[Path, str, str]:
        d = ensure_project_dir(project_id)
        dieline_input = d / "dieline_input.png"
        concept_input = d / payload.get("concept_file", "package_input.png")
        output_path = d / "dieline_result.png"

        if not dieline_input.exists():
            raise FileNotFoundError(f"Missing: {dieline_input}")
        if not concept_input.exists():
            raise FileNotFoundError(f"Missing: {concept_input}")

        gen = DielineGenerator(api_key=self.api_key)
        gen.generate(
            dieline_path=str(dieline_input),
            concept_path=str(concept_input),
            output_path=str(output_path),
        )
        return output_path, f"{project_id}/dieline_result.png", "image/png"

    async def run_video(self, project_id: int, payload: Dict[str, Any]) -> Tuple[Path, str, str]:
        d = ensure_project_dir(project_id)
        package_path = d / "package.png"
        if not package_path.exists():
            raise FileNotFoundError("package.png not found. VIDEO requires package.png in ai/{projectId}/")

        with open(package_path, "rb") as f:
            file_bytes = f.read()

        out = await asyncio.wait_for(
            generate_video_for_product(product_id=project_id, req=payload, product_image=file_bytes),
            timeout=self.video_timeout_sec,
        )
        out_path = Path(out)
        return out_path, f"{project_id}/video.mp4", "video/mp4"

    # ✅ BASIC: (기존 PACKAGE) package_input.png + prompt -> package.png 생성 후 업로드
    def run_basic(self, project_id: int, payload: Dict[str, Any]) -> Tuple[Path, str, str]:
        d = ensure_project_dir(project_id)
        input_path = d / "package_input.png"
        output_path = d / "package.png"

        if not input_path.exists():
            raise FileNotFoundError(f"{input_path} not found (BASIC requires package_input.png)")

        prompt = payload["prompt"]

        generator = PackageGenerator()
        # 서비스 구현에 맞춰 호출 (너 기존 코드 기준)
        generator.edit_package_image(product_dir=d, instruction=prompt)

        if not output_path.exists():
            raise RuntimeError("package.png was not generated")

        return output_path, f"{project_id}/package.png", "image/png"

    async def execute(self, job: Dict[str, Any]) -> Dict[str, Any]:
        job_type = str(job.get("jobType", "")).upper()
        project_id = int(job.get("projectId", 0) or 0)
        job_id = str(job.get("jobId") or "").strip()
        if not job_id:
            raise ValueError("jobId is empty")

        payload = job.get("payload") or {}
        if not isinstance(payload, dict):
            raise ValueError(f"payload must be object, got {type(payload)}")

        payload_norm = normalize_payload(job_type, payload)

        # ✅ BASIC도 projectId 필수 (패키지 생성이니까)
        if job_type == "BASIC":
            if project_id <= 0:
                raise ValueError("projectId is required for BASIC")
            local_path, obj, ct = self.run_basic(project_id, payload_norm)
            gs_uri, url = self.uploader.upload_file(local_path, obj, content_type=ct)
            return {"outputUri": gs_uri, "outputUrl": url}

        # 나머지도 projectId 필수
        if project_id <= 0:
            raise ValueError("projectId is required for non-BASIC jobs")

        if job_type == "BANNER":
            local_path, obj, ct = self.run_banner(project_id, payload_norm)
            gs_uri, url = self.uploader.upload_file(local_path, obj, content_type=ct)
            return {"outputUri": gs_uri, "outputUrl": url}

        if job_type == "SNS":
            local_path, obj, ct = self.run_sns(project_id, payload_norm)
            gs_uri, url = self.uploader.upload_file(local_path, obj, content_type=ct)
            return {"outputUri": gs_uri, "outputUrl": url}

        if job_type == "VIDEO":
            local_path, obj, ct = await self.run_video(project_id, payload_norm)
            gs_uri, url = self.uploader.upload_file(local_path, obj, content_type=ct)
            return {"outputUri": gs_uri, "outputUrl": url}

        if job_type == "DIELINE":
            local_path, obj, ct = self.run_dieline(project_id, payload_norm)
            gs_uri, url = self.uploader.upload_file(local_path, obj, content_type=ct)
            return {"outputUri": gs_uri, "outputUrl": url}

        raise ValueError(f"unsupported jobType: {job_type}")


# =========================
# Result publisher
# =========================
class ResultPublisher:
    def __init__(self, channel: aio_pika.Channel, result_queue: str):
        self.channel = channel
        self.result_queue = result_queue

    async def publish(self, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        msg = Message(body=body, content_type="application/json", delivery_mode=DeliveryMode.PERSISTENT)
        await self.channel.default_exchange.publish(msg, routing_key=self.result_queue)


# =========================
# Consumer
# =========================
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

            jt = str(job.get("jobType", "")).upper()
            pid = job.get("projectId")
            print(f"[JOB START] jobId={job_id} type={jt} projectId={pid}", flush=True)

            out = await runner.execute(job)

            await pub.publish(
                {
                    "jobId": job_id,
                    "success": True,
                    "outputUri": out.get("outputUri"),
                    "outputUrl": out.get("outputUrl"),
                    "errorCode": None,
                    "errorMessage": None,
                }
            )

            print(json.dumps({"ok": True, "jobId": job_id, "output": out}, ensure_ascii=False), flush=True)

        except Exception as e:
            print("[JOB ERROR] job failed", flush=True)
            traceback.print_exc()

            try:
                await pub.publish(
                    {
                        "jobId": job_id,
                        "success": False,
                        "outputUri": None,
                        "outputUrl": None,
                        "errorCode": "WORKER_FAILED",
                        "errorMessage": str(e) if e else "unknown error",
                    }
                )
            except Exception:
                print("[RESULT PUBLISH ERROR] failed to publish job result", flush=True)
                traceback.print_exc()


async def main():
    env = load_env()
    if not env.rabbitmq_url:
        raise ValueError("RABBITMQ_URL is empty")
    if not env.queue_jobs:
        raise ValueError("RABBITMQ_QUEUE is empty")
    if not env.gcs_bucket:
        raise ValueError("GCS_BUCKET is empty")
    if not env.gcs_public_base_url:
        raise ValueError("GCS_PUBLIC_BASE_URL is empty")

    print(f"Starting consumer: url={env.rabbitmq_url} jobs={env.queue_jobs} results={env.queue_results}", flush=True)

    connection = await aio_pika.connect_robust(env.rabbitmq_url)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=10)

    await channel.declare_queue(env.queue_jobs, durable=True)
    await channel.declare_queue(env.queue_results, durable=True)

    uploader = GcsUploader(env.gcs_bucket, env.gcs_public_base_url)
    runner = JobRunner(env.api_key, uploader, video_timeout_sec=env.video_timeout_sec)
    publisher = ResultPublisher(channel, env.queue_results)

    q = await channel.get_queue(env.queue_jobs)
    await q.consume(lambda m: handle_message(m, runner, publisher))

    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
