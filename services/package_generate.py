import io
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types

GEMINI_API_KEY = "AIzaSyDIQ0AoxpM8i14ZZ-x6Ey_VdmWyXJsLw5I"
GEMINI_MODEL = "gemini-3-pro-image-preview"


class PackageGenerator:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def edit_package_image(
        self,
        product_dir: Path,
        instruction: str,
        resize_long_side_px: int = 1536,
    ) -> str:

        if not instruction or not instruction.strip():
            raise ValueError("instruction is empty")

        product_dir = Path(product_dir).resolve()
        input_path = (product_dir / "package_input.png").resolve()
        output_path = (product_dir / "package.png").resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"input image not found: {input_path}")

        original = Image.open(input_path).convert("RGBA")
        original = self._resize_if_needed(original, resize_long_side_px)

        prompt = f"""
ACT AS: Expert Package Design Editor.
TASK: Edit the provided image strictly according to the USER INSTRUCTION.

CRITICAL RULES:
1) OUTPUT: Return an edited image (not text).
2) PRESERVATION: Keep original Brand Logo and Product Name text intact.
3) CONSISTENCY: Maintain the main character identity while applying changes.

USER INSTRUCTION:
{instruction}
"""

        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, original],
            config=types.GenerateContentConfig(),
        )

        edited_img = None
        if response and getattr(response, "parts", None):
            for part in response.parts:
                inline = getattr(part, "inline_data", None)
                if inline and inline.data:
                    edited_img = Image.open(io.BytesIO(inline.data)).convert("RGBA")
                    break

        if edited_img is None:
            raise RuntimeError("Model did not return image")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        edited_img.save(output_path, format="PNG")

        return str(output_path)

    def _resize_if_needed(self, img: Image.Image, limit_px: int) -> Image.Image:
        w, h = img.size
        long_side = max(w, h)
        if long_side <= limit_px:
            return img
        scale = limit_px / float(long_side)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return img.resize((new_w, new_h), Image.LANCZOS)
