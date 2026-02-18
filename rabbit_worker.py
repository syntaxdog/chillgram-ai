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
from services.package_generate import PackageGenerator

load_dotenv()

# [Fix] rembg U2NET_HOME permission denied error
# 컨테이너 --user 1005:1006 실행 시 HOME=/ → /.u2net 권한 에러 방지
# Dockerfile ENV U2NET_HOME과 이중 안전장치
os.environ.setdefault("U2NET_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".u2net"))

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
    prefetch_count: int
    worker_concurrency: int


def load_env() -> Env:
    bucket = os.getenv("GCS_BUCKET", "").strip()
    public_base = os.getenv("GCS_PUBLIC_BASE_URL", "").strip()
    if not public_base and bucket:
        public_base = f"https://storage.googleapis.com/{bucket}"

    worker_conc = int(os.getenv("WORKER_CONCURRENCY", "4"))
    prefetch = int(os.getenv("RABBITMQ_PREFETCH", str(max(worker_conc, worker_conc * 2))))

    return Env(
        api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        rabbitmq_url=os.getenv("RABBITMQ_URL", "").strip(),
        queue_jobs=os.getenv("RABBITMQ_QUEUE", "").strip(),
        queue_results=os.getenv("RABBITMQ_RESULT_QUEUE", "chillgram.job-results").strip(),
        gcs_bucket=bucket,
        video_timeout_sec=int(os.getenv("VIDEO_TIMEOUT_SEC", "900")),
        gcs_public_base_url=public_base,
        prefetch_count=prefetch,
        worker_concurrency=worker_conc,
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
        # headline 제거 (사용 안함)
        typo_text = pick(payload, "typo_text", "typoText", "finalCopy")
        ratio = pick(payload, "ratio", "Ratio", "bannerRatio", "bannerSize")
        guideline = pick(payload, "guideline", "Guideline", "guideLine", default="")
        if isinstance(guideline, dict):
            guideline = json.dumps(guideline, ensure_ascii=False)

        base_image_url = pick(payload, "baseImageUrl", "base_image_url")

        if not typo_text:
            raise ValueError(f"BANNER payload 필수값 누락: typoText (Available keys: {list(payload.keys())})")
        if not ratio:
            raise ValueError(f"BANNER payload 필수값 누락: ratio (Available keys: {list(payload.keys())})")
        
        return {
            "typo_text": typo_text, 
            "ratio": ratio, 
            "guideline": guideline,
            "baseImageUrl": base_image_url
        }

    if jt == "SNS":
        main_text = pick(payload, "main_text", "mainText", "finalCopy", "typoText")
        guideline = pick(payload, "guideline", "guideLine")
        base_image_url = pick(payload, "baseImageUrl", "base_image_url")

        sub_text = pick(payload, "sub_text", "subText", default="")
        preset = pick(payload, "preset")
        custom_prompt = pick(payload, "custom_prompt", "customPrompt")
        save_background = pick(payload, "save_background", "saveBackground", default=True)

        if not main_text:
            raise ValueError(f"SNS payload missing: mainText/main_text (Available keys: {list(payload.keys())})")
        return {
            "main_text": main_text,
            "sub_text": sub_text or "",
            "preset": preset,
            "custom_prompt": custom_prompt,
            "guideline": guideline,
            "save_background": bool(save_background),
            "baseImageUrl": base_image_url
        }

    if jt == "VIDEO":
        food_name = pick(payload, "food_name", "foodName")
        food_type = pick(payload, "food_type", "foodType")
        ad_concept = pick(payload, "ad_concept", "adConcept")
        ad_req = pick(payload, "ad_req", "adReq")
        
        # [수정] 필수값이 없더라도, 이미지가 있으면 video_2.py에서 자동 추론하도록 허용
        # 최소한의 식별자(baseImageUrl)는 있어야 함 (또는 이미 다운로드된 package.png)
        # return raw payload keys if missing, so video_2.py handles defaults
        return {
            "food_name": food_name, 
            "food_type": food_type, 
            "ad_concept": ad_concept, 
            "ad_req": ad_req,
            "baseImageUrl": pick(payload, "baseImageUrl", "base_image_url") # 이미지 URL 전달
        }

    if jt == "DIELINE":
        prompt = pick(payload, "prompt", "instruction", default="")
        concept_file = pick(payload, "concept_file", "conceptFile", default="package_input.png")
        return {"prompt": prompt, "concept_file": concept_file}

    sub_type = pick(payload, "subType", "sub_type")
    if sub_type == "DIELINE":
        input_url = pick(payload, "inputUrl", "input_url", "inputURI", "inputUri")
        prompt = pick(payload, "prompt", "instruction", default="")
        concept_url = pick(payload, "conceptUrl", "concept_url")
        return {
            "subType": "DIELINE",
            "inputUrl": input_url,
            "prompt": prompt,
            "conceptUrl": concept_url,
        }

    if jt == "BASIC":
        input_url = pick(payload, "inputUrl", "input_url", "inputURI", "inputUri")
        prompt = pick(payload, "prompt", "instruction", default="")
        if not input_url:
            raise ValueError("BASIC payload missing: inputUrl")
        return {"inputUrl": input_url, "prompt": prompt}

    return payload


# =========================
# GCS helpers
# =========================
def parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
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


def parse_gcs_http_url(url: str) -> Tuple[str, str]:
    """
    Supports:
    - https://storage.googleapis.com/<bucket>/<object>
    - https://<bucket>.storage.googleapis.com/<object>
    """
    from urllib.parse import urlparse

    u = url.strip()
    p = urlparse(u)
    host = p.netloc
    path = p.path or ""

    if host == "storage.googleapis.com":
        # /bucket/object
        if not path.startswith("/") or len(path) <= 1:
            raise ValueError(f"invalid gcs http url: {url}")
        rest = path[1:]
        slash = rest.find("/")
        if slash < 0:
            raise ValueError(f"invalid gcs http url: {url}")
        return rest[:slash], rest[slash + 1 :]

    suffix = ".storage.googleapis.com"
    if host.endswith(suffix):
        bucket = host[: -len(suffix)]
        obj = path[1:] if path.startswith("/") else path
        if not obj:
            raise ValueError(f"invalid gcs http url (empty object): {url}")
        return bucket, obj

    raise ValueError(f"unsupported gcs http url host: {host}")


class GcsUploader:
    def __init__(self, bucket: str, public_base_url: str):
        if not bucket:
            raise ValueError("GCS_BUCKET is empty")
        if not public_base_url:
            raise ValueError("GCS_PUBLIC_BASE_URL is empty")
        self.bucket_name = bucket
        self.public_base_url = public_base_url.rstrip("/")
        self.client = storage.Client()

    def upload_file(self, local_path: Path, object_name: str, content_type: Optional[str] = None) -> str:
        """
        ✅ gs://는 외부로 내보내지 않는다.
        리턴: public https url
        """
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(object_name)
        if content_type:
            blob.content_type = content_type
        blob.upload_from_filename(str(local_path))
        public_url = f"{self.public_base_url}/{object_name}"
        return public_url

    def download_to_file(self, uri: str, dest: Path) -> None:
        """
        ✅ inputUrl이 https여도 다운 가능하게.
        - gs://bucket/object
        - https://storage.googleapis.com/bucket/object
        - https://bucket.storage.googleapis.com/object
        """
        u = uri.strip()
        if u.startswith("gs://"):
            bkt, obj = parse_gs_uri(u)
        elif u.startswith("http://") or u.startswith("https://"):
            bkt, obj = parse_gcs_http_url(u)
        else:
            raise ValueError(f"unsupported inputUrl scheme: {uri}")

        blob = self.client.bucket(bkt).blob(obj)
        if not blob.exists():
            raise FileNotFoundError(f"gcs object not found: {uri}")

        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))


# =========================
# Runner
# =========================
class JobRunner:
    def __init__(self, api_key: str, uploader: GcsUploader, video_timeout_sec: int):
        self.api_key = api_key
        self.uploader = uploader
        self.video_timeout_sec = video_timeout_sec

    # ---- sync runners (원본) ----
    def run_banner(self, project_id: int, payload: Dict[str, Any]) -> Tuple[Path, str, str]:
        d = ensure_project_dir(project_id)
        input_path = d / "package.png"

        # ✅ baseImageUrl 다운로드 로직 추가
        base_image_url = payload.get("baseImageUrl")
        if not input_path.exists() and base_image_url:
            print(f"[BANNER] Downloading base image from {base_image_url}", flush=True)
            self.uploader.download_to_file(base_image_url, input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"{input_path} not found (package.png required)")

        gen = AdBannerGenerator(api_key=self.api_key, ratio=payload["ratio"])

        filename = "banner.png"
        output_path = d / filename

        gen.process(
            image_path=str(input_path),
            output_path=str(output_path),
            typo_text=payload["typo_text"],
            guideline=payload.get("guideline", ""),
        )
        return output_path, f"{project_id}/{filename}", "image/png"

    def run_sns(self, project_id: int, payload: Dict[str, Any]) -> Tuple[Path, str, str]:
        d = ensure_project_dir(project_id)
        product_path = d / "package.png"

        # ✅ baseImageUrl에서 항상 다운로드
        base_image_url = payload.get("baseImageUrl")
        if not base_image_url:
            raise ValueError("SNS payload missing: baseImageUrl")
        print(f"[SNS] Downloading base image from {base_image_url}", flush=True)
        self.uploader.download_to_file(base_image_url, product_path)

        background_path = d / "sns_background.png"
        final_path = d / "sns.png"

        gen = SNSImageGenerator(api_key=self.api_key)
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

    def run_dieline(self, project_id: int, payload: Dict[str, Any], job_id: Optional[str] = None) -> Tuple[Path, str, str]:
        """
        job_id가 있으면(미리보기/Basic) -> ai/tmp/package/{job_id}/ 사용
        job_id가 없으면(프로젝트 저장) -> ai/{project_id}/ 사용
        """
        if job_id:
            d = ensure_dir(AI_DIR / "tmp" / "package" / job_id)
            dieline_input = d / "dieline_input.png"
            concept_input = d / "concept_input.png"
            output_path = d / "package.png"  # BASIC은 package.png로 리턴 기대

            input_uri = payload.get("inputUrl")
            if input_uri:
                self.uploader.download_to_file(input_uri, dieline_input)

            concept_uri = payload.get("conceptUrl")
            if concept_uri:
                self.uploader.download_to_file(concept_uri, concept_input)

            object_name_suffix = f"tmp/package/{job_id}/package.png"
        else:
            d = ensure_project_dir(project_id)
            dieline_input = d / "dieline_input.png"
            concept_input = d / payload.get("concept_file", "package_input.png")
            output_path = d / "dieline_result.png"
            object_name_suffix = f"{project_id}/dieline_result.png"

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
        return output_path, object_name_suffix, "image/png"

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

    def run_basic_preview(self, job_id: str, payload: Dict[str, Any]) -> str:
        input_uri = payload["inputUrl"]
        prompt = payload["prompt"]

        tmp_dir = ensure_dir(AI_DIR / "tmp" / "package" / job_id)
        output_path = tmp_dir / "package.png"

        input_path = tmp_dir / "package_input.png"
        # 1) inputUrl 다운로드 (gs/http 모두 지원)
        self.uploader.download_to_file(input_uri, input_path)

        # 2) 패키지 생성/편집
        generator = PackageGenerator()
        generator.edit_package_image(product_dir=tmp_dir, instruction=prompt)
        if not output_path.exists():
            raise RuntimeError("package.png was not generated")

        # 3) GCS 업로드 (✅ public https url만 리턴)
        object_name = f"tmp/package/{job_id}/package.png"
        url = self.uploader.upload_file(output_path, object_name, content_type="image/png")
        return url

    # ---- async wrappers: blocking 작업을 thread로 ----
    async def _to_thread(self, fn, *args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def upload_file_async(self, local_path: Path, object_name: str, content_type: Optional[str] = None) -> str:
        return await self._to_thread(self.uploader.upload_file, local_path, object_name, content_type)

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

        # ✅ (수정) DIELINE subType 우선 인터셉트 (BASIC 로직 오염 방지)
        if payload_norm.get("subType") == "DIELINE":
            # run_dieline은 blocking + GCS download 포함 -> thread
            out_path, obj_suffix, ct = await self._to_thread(self.run_dieline, project_id, payload_norm, job_id)
            url = await self.upload_file_async(out_path, obj_suffix, content_type=ct)
            return {"outputUri": url}

        # ✅ BASIC: projectId=0 OK (outputUri = https)
        if job_type == "BASIC":
            url = await self._to_thread(self.run_basic_preview, job_id, payload_norm)
            return {"outputUri": url}

        # 나머지는 projectId 필수
        if project_id <= 0:
            raise ValueError("projectId is required for non-BASIC jobs")

        if job_type == "BANNER":
            local_path, obj, ct = await self._to_thread(self.run_banner, project_id, payload_norm)
            url = await self.upload_file_async(local_path, obj, content_type=ct)
            return {"outputUri": url}

        if job_type == "SNS":
            local_path, obj, ct = await self._to_thread(self.run_sns, project_id, payload_norm)
            url = await self.upload_file_async(local_path, obj, content_type=ct)
            return {"outputUri": url}

        if job_type == "VIDEO":
            local_path, obj, ct = await self.run_video(project_id, payload_norm)
            url = await self.upload_file_async(local_path, obj, content_type=ct)
            return {"outputUri": url}

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
async def handle_message(msg: IncomingMessage, runner: JobRunner, pub: ResultPublisher, sem: asyncio.Semaphore):
    raw = msg.body.decode("utf-8", errors="replace")
    print(f"[RECEIVED] {raw}", flush=True)

    # ✅ 동시 처리량 제한
    async with sem:
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

                # ✅ outputUri만 보냄 (gs:// 제거)
                await pub.publish(
                    {
                        "jobId": job_id,
                        "success": True,
                        "outputUri": out.get("outputUri"),
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

    print(
        f"Starting consumer: url={env.rabbitmq_url} jobs={env.queue_jobs} results={env.queue_results} "
        f"prefetch={env.prefetch_count} concurrency={env.worker_concurrency}",
        flush=True,
    )

    connection = await aio_pika.connect_robust(env.rabbitmq_url)
    channel = await connection.channel()

    # ✅ prefetch는 동시 처리량(sem)과 비슷하게/약간 크게
    await channel.set_qos(prefetch_count=env.prefetch_count)

    await channel.declare_queue(env.queue_jobs, durable=True)
    await channel.declare_queue(env.queue_results, durable=True)

    uploader = GcsUploader(env.gcs_bucket, env.gcs_public_base_url)
    runner = JobRunner(env.api_key, uploader, video_timeout_sec=env.video_timeout_sec)
    publisher = ResultPublisher(channel, env.queue_results)

    sem = asyncio.Semaphore(env.worker_concurrency)

    q = await channel.get_queue(env.queue_jobs)
    await q.consume(lambda m: handle_message(m, runner, publisher, sem))

    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
