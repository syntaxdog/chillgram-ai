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

        # --- [수정된 부분: 수량 제한(Single Object) 규칙 추가] ---
        prompt = f"""
ACT AS: Expert Product Designer and Photo Retoucher.
TASK: Redesign the graphics ON the package while keeping the exact original package shape, then isolate it on a white background.

CRITICAL RULES (FOLLOW IN ORDER):
1. STRUCTURAL INTEGRITY (SHAPE & ORIENTATION) [MISSING PART ADDED]:
   - The physical shape, outer contour, and ASPECT RATIO (height vs. width) of the package MUST remain 100% identical to the original input image.
   - If the original is VERTICAL (Portrait), the output MUST be VERTICAL. Do NOT rotate, widen, or change the form factor.
   - Treat the original package silhouette as a fixed canvas that cannot be resized.

2. GRAPHIC REDESIGN (PRIMARY GOAL): 
   - Dramatically change the graphics, colors, patterns, and background illustrations *ON* the product package surface according to the USER INSTRUCTION.

3. UNIVERSAL PRESERVATION (SMART DETECTION): 
   - Automatically IDENTIFY the **Main Brand Logo** and the **Primary Product Name**.
   - These elements MUST remain legible, intact, and in their original relative positions.
   - Do NOT change the text content or the logo shape.

4. QUANTITY & COMPOSITION:
   - Output EXACTLY ONE (1) single product package.
   - Place it in the CENTER.

5. ISOLATION & REMOVAL: 
   - Remove the entire environmental background AND the hand/fingers. 
   - Reconstruct hidden areas seamlessly.

6. NEW BACKGROUND: 
   - Place the package on a clean, solid WHITE background (hex #FFFFFF).

USER INSTRUCTION:
{instruction}
"""
        # ------------------------------------------------------------------

        # 창의성과 규칙 준수의 밸런스를 위해 0.6~0.7 유지
        generation_config = types.GenerateContentConfig(
            temperature=0.7
        )

        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, original],
            config=generation_config,
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
