# services/sns_image_generate.py


import os
import sys
import io
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter, ImageStat, ImageEnhance

# Windows 콘솔 이슈 대응
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    from google import genai
    from google.genai import types
except ImportError as e:
    raise ImportError("google-genai 필요: pip install google-genai") from e

try:
    import cv2
except ImportError as e:
    raise ImportError("opencv-python 필요: pip install opencv-python") from e

try:
    from rembg import remove as remove_bg

    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False
    # rembg 없어도 돌아가게 유지
    print("Warning: rembg 없음 - 배경 제거 기능 비활성화")

# dotenv는 있으면 로드
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# =========================
# Constants
# =========================
INSTAGRAM_SIZE = (1080, 1350)
DEFAULT_PRODUCT_SCALE = 0.50
TONE_BLEND_STRENGTH = 0.15  # 톤 매칭 강도 (0.0 ~ 1.0)

PRODUCT_SPACE_INSTRUCTION = """
CRITICAL LAYOUT REQUIREMENT:
- Leave the CENTER AREA of the image COMPLETELY EMPTY for product placement
- DO NOT draw any product, package, or object in the center region
- Props and decorative elements should only be at the EDGES
- The center 40% of the image must be clean and unobstructed
- This empty space will be used to overlay the actual product later
"""

PRESETS = {
    "ocean_sunset": """Create a premium ocean sunset food photography background:
- Beautiful ocean/sea view with warm golden sunset in the upper portion
- Rustic wooden table surface in the lower foreground
- Food styling props NEATLY ARRANGED on the table (not floating):
  * A few whole shrimp lying naturally on the table surface
  * Small pile of sea salt
  * Scattered peppercorns or herbs
- Props should be at the EDGES only, leave center empty
- IMPORTANT: Leave the CENTER AREA COMPLETELY EMPTY for product placement
- Warm, appetizing lighting with golden hour tones
- Professional food photography style, vertical format (4:5 ratio)""",
    "sports_stadium": """Create a dramatic sports advertisement background:
- Football stadium at night with bright lights and dramatic light rays
- Excited crowd silhouettes cheering with raised hands in the background
- Fireworks, confetti, and golden spark effects in the sky
- IMPORTANT: Leave the CENTER AREA EMPTY for product placement later
- DO NOT draw any product package - only the background environment
- Keep the center clean - props only at the very bottom edges if any
- Epic cinematic lighting with orange/golden tones, 8k ultra detailed
- Professional SNS advertisement style, vertical format (9:16 ratio)""",
    "fire_explosion": """Create an intense fire-themed advertisement background:
- Dramatic fire and bright sparks on dark/black background
- Intense orange and red flames as background effect
- IMPORTANT: Leave the CENTER AREA COMPLETELY EMPTY for product placement
- DO NOT draw any product or floating objects in the center
- Fire effects should frame the edges, not fill the center
- Cinematic dramatic lighting, 8k quality, high contrast
- Professional advertisement poster style, vertical format""",
    "luxury_premium": """Create a premium luxury advertisement background:
- Elegant dark marble or granite surface
- Soft golden rim lighting from the sides
- Subtle sparkles and premium bokeh effects
- IMPORTANT: Leave the CENTER AREA EMPTY for product placement
- High-end product photography lighting setup
- Professional luxury brand advertisement style, vertical format""",
}


class SNSImageGenerator:
    """Gemini 기반 SNS 광고 이미지 생성기"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY 필요 (환경변수로 설정하거나 생성자에 전달)"
            )
        self.client = genai.Client(api_key=self.api_key)

    def generate_background(
        self,
        product_image: Image.Image,
        main_text: str,
        sub_text: str = "",
        preset: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        guideline: Optional[str] = None,
    ) -> Image.Image:
        """Gemini로 배경 + 타이포그래피 생성

        Args:
            product_image: 제품 이미지
            main_text: 메인 텍스트
            sub_text: 서브 텍스트
            preset: 프리셋 이름
            custom_prompt: 커스텀 프롬프트
            guideline: 다른 팀에서 제공하는 필수 가이드라인 프롬프트
        """

        # 프롬프트 우선순위: guideline > custom_prompt > preset
        if guideline:
            base_prompt = guideline
            print("  Using team guideline (with auto product-space and typography instruction)")
        elif custom_prompt:
            base_prompt = custom_prompt
            print("  Using custom prompt (with auto product-space and typography instruction)")
        else:
            preset = preset or "ocean_sunset"
            base_prompt = PRESETS.get(preset, PRESETS["ocean_sunset"])
            print(f"  Generating background (preset: {preset})")

        # 타이포그래피 지침 (필수)
        typography_instruction = f"""
TYPOGRAPHY REQUIREMENTS:
- Add elegant text at the BOTTOM of the image (lower 15% area)
- Main text: "{main_text}" in premium white/golden Korean typography
- {"Subtitle: " + sub_text + " below main text in smaller white font" if sub_text else ""}
- Text should be centered, with subtle shadow for readability
- Make typography look premium and high-end
- Modern, clean Korean font style"""

        # guideline 또는 custom_prompt 사용 시: base_prompt + 필수 지침들
        if guideline or custom_prompt:
            prompt = f"{base_prompt}{PRODUCT_SPACE_INSTRUCTION}{typography_instruction}"
        else:
            # preset 사용 시: preset + 타이포 + 추가 안내
            prompt = f"""{base_prompt}
{typography_instruction}

IMPORTANT:
- Reference this product image to understand the product style/colors
- Generate matching food props scattered around edges"""

        response = self.client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[prompt, product_image],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                temperature=0.9,
                image_config=types.ImageConfig(
                    aspect_ratio="4:5",
                    image_size="2K",
                ),
            ),
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return Image.open(io.BytesIO(part.inline_data.data))

        raise RuntimeError("Gemini 배경 생성 실패")

    def analyze_lighting(self, background: Image.Image) -> dict:
        """배경 이미지의 조명 방향과 색상 분석"""
        bg_array = np.array(background.convert("RGB"))
        h, w = bg_array.shape[:2]

        quadrants = {
            "top_left": bg_array[: h // 2, : w // 2],
            "top_right": bg_array[: h // 2, w // 2 :],
            "bottom_left": bg_array[h // 2 :, : w // 2],
            "bottom_right": bg_array[h // 2 :, w // 2 :],
        }

        brightness = {name: np.mean(quad) for name, quad in quadrants.items()}

        top_avg = (brightness["top_left"] + brightness["top_right"]) / 2
        bottom_avg = (brightness["bottom_left"] + brightness["bottom_right"]) / 2
        left_avg = (brightness["top_left"] + brightness["bottom_left"]) / 2
        right_avg = (brightness["top_right"] + brightness["bottom_right"]) / 2

        light_y = "top" if top_avg > bottom_avg else "bottom"
        light_x = "right" if right_avg > left_avg else "left"

        brightest_quad = max(quadrants.keys(), key=lambda k: brightness[k])
        rim_color = tuple(np.mean(quadrants[brightest_quad], axis=(0, 1)).astype(int))

        return {
            "direction": (light_x, light_y),
            "rim_color": rim_color,
        }

    def match_tone(
        self,
        product: Image.Image,
        background: Image.Image,
        strength: float = TONE_BLEND_STRENGTH,
    ) -> Image.Image:
        """배경과 상품의 색조/톤 매칭"""
        alpha = product.split()[3]
        product_rgb = product.convert("RGB")

        cx, cy = background.width // 2, background.height // 2
        crop_size = min(background.width, background.height) // 3
        center_crop = background.crop(
            (
                cx - crop_size,
                cy - crop_size,
                cx + crop_size,
                cy + crop_size,
            )
        ).convert("RGB")

        bg_stat = ImageStat.Stat(center_crop)
        prod_stat = ImageStat.Stat(product_rgb)

        bg_mean = bg_stat.mean
        prod_mean = prod_stat.mean

        bg_brightness = sum(bg_mean) / 3
        prod_brightness = sum(prod_mean) / 3

        if prod_brightness > 0:
            brightness_ratio = bg_brightness / prod_brightness
            brightness_ratio = (
                0.7 + (brightness_ratio - 0.7) * strength + 0.3 * (1 - strength)
            )
            brightness_ratio = max(0.8, min(1.2, brightness_ratio))
            enhancer = ImageEnhance.Brightness(product_rgb)
            product_rgb = enhancer.enhance(brightness_ratio)

        tint_layer = Image.new("RGB", product_rgb.size, tuple(int(c) for c in bg_mean))
        product_rgb = Image.blend(product_rgb, tint_layer, strength * 0.5)

        result = product_rgb.convert("RGBA")
        result.putalpha(alpha)

        print(
            f"  Tone matched (bg: {bg_brightness:.0f}, product: {prod_brightness:.0f})"
        )
        return result

    def overlay_product(
        self,
        background: Image.Image,
        product: Image.Image,
        scale: float = DEFAULT_PRODUCT_SCALE,
    ) -> Image.Image:
        """배경 위에 상품 오버레이 (그림자 + 림라이트 효과)"""
        result = background.copy().convert("RGBA")

        lighting = self.analyze_lighting(background)
        light_x, light_y = lighting["direction"]
        rim_color = lighting["rim_color"]
        print(f"  Light: {light_x}-{light_y}, rim color: {rim_color}")

        if HAS_REMBG:
            product_rgba = remove_bg(product).convert("RGBA")
            print("  Background removed")
        else:
            product_rgba = product.convert("RGBA")

        target_h = int(result.height * scale)
        ratio = target_h / product_rgba.height
        target_w = int(product_rgba.width * ratio)
        product_resized = product_rgba.resize((target_w, target_h), Image.LANCZOS)

        product_resized = self.match_tone(product_resized, result)

        x = (result.width - target_w) // 2
        y = (result.height - target_h) // 2 - int(result.height * 0.08)
        print(f"  Product: {target_w}x{target_h}, position: ({x}, {y})")

        prod_array = np.array(product_resized)

        # === 1) 바닥 그림자 ===
        shadow_offset_x = 8 if light_x == "left" else -8
        ground_shadow = Image.new("RGBA", result.size, (0, 0, 0, 0))
        ground_array = np.array(ground_shadow)

        shadow_w = int(target_w * 0.75)
        shadow_h = int(target_h * 0.12)
        shadow_cx = x + target_w // 2 + shadow_offset_x
        shadow_cy = y + target_h - int(shadow_h * 0.3)

        for sy in range(
            max(0, shadow_cy - shadow_h), min(result.height, shadow_cy + shadow_h)
        ):
            for sx in range(
                max(0, shadow_cx - shadow_w), min(result.width, shadow_cx + shadow_w)
            ):
                dx = (sx - shadow_cx) / shadow_w if shadow_w > 0 else 0
                dy = (sy - shadow_cy) / (shadow_h * 0.5) if shadow_h > 0 else 0
                dist = (dx**2 + dy**2) ** 0.5
                if dist < 1.0:
                    falloff = np.exp(-3 * dist**2)
                    a = int(falloff * 90)
                    ground_array[sy, sx, 3] = max(ground_array[sy, sx, 3], a)

        ground_alpha = ground_array[:, :, 3].astype(np.float32)
        ground_alpha = cv2.GaussianBlur(ground_alpha, (0, 0), 25)
        ground_array[:, :, 3] = ground_alpha.astype(np.uint8)
        ground_shadow = Image.fromarray(ground_array)
        result = Image.alpha_composite(result, ground_shadow)

        # === 2) 림 라이트 ===
        if prod_array.shape[2] == 4:
            rim_img = Image.new("RGBA", product_resized.size, (0, 0, 0, 0))
            rim_array = np.array(rim_img)

            alpha_2d = prod_array[:, :, 3].astype(np.float32)
            sobel_x = cv2.Sobel(alpha_2d, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(alpha_2d, cv2.CV_64F, 0, 1, ksize=3)
            edge_mag = np.sqrt(sobel_x**2 + sobel_y**2)

            if edge_mag.max() > 0:
                edge_mag = (edge_mag / edge_mag.max() * 255).astype(np.uint8)

            rim_intensity = 0.4
            for py in range(prod_array.shape[0]):
                for px in range(prod_array.shape[1]):
                    if edge_mag[py, px] > 30:
                        facing_light = False
                        if light_x == "right" and px > target_w * 0.5:
                            facing_light = True
                        elif light_x == "left" and px < target_w * 0.5:
                            facing_light = True
                        if light_y == "top" and py < target_h * 0.4:
                            facing_light = True

                        if facing_light:
                            intensity = min(edge_mag[py, px] / 255 * rim_intensity, 1.0)
                            rim_array[py, px, 0] = int(rim_color[0] * intensity)
                            rim_array[py, px, 1] = int(rim_color[1] * intensity)
                            rim_array[py, px, 2] = int(rim_color[2] * intensity)
                            rim_array[py, px, 3] = int(100 * intensity)

            rim_img = Image.fromarray(rim_array)
            rim_img = rim_img.filter(ImageFilter.GaussianBlur(radius=2))
            product_with_rim = Image.alpha_composite(product_resized, rim_img)
            result.paste(product_with_rim, (x, y), product_with_rim)
        else:
            result.paste(product_resized, (x, y), product_resized)

        return result

    def generate(
        self,
        product_path: str,
        main_text: str = "광고 문구",
        sub_text: str = "",
        preset: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        guideline: Optional[str] = None,
        output_path: Optional[str] = None,
        save_background: bool = False,
        background_output_path: Optional[str] = None,
    ) -> Image.Image:
        """SNS 광고 이미지 생성 (FastAPI service용)

        Args:
            product_path: 상품 이미지 경로
            main_text: 메인 텍스트
            sub_text: 서브 텍스트
            preset: 프리셋 이름 (custom_prompt 없을 때 사용)
            custom_prompt: 커스텀 프롬프트 (있으면 preset 무시)
            guideline: 다른 팀에서 제공하는 필수 가이드라인 프롬프트
            output_path: 최종 결과 저장 경로 (sns.png)
            save_background: 배경 이미지 별도 저장 여부
            background_output_path: 배경 저장 경로 (sns_background.png)
        """
        print("=" * 50)
        print("  SNS Image Generator")
        print("=" * 50)

        # 상품 로드
        print("\n[1/3] Loading product...")
        product = Image.open(product_path)
        print(f"  Size: {product.size}")

        # 배경 생성
        print("\n[2/3] Generating background + typography...")
        background = self.generate_background(
            product, main_text, sub_text, preset=preset, custom_prompt=custom_prompt, guideline=guideline
        )
        print(f"  Background: {background.size}")

        # 배경 저장
        if save_background and background_output_path:
            Path(background_output_path).parent.mkdir(parents=True, exist_ok=True)
            background.save(background_output_path)
            print(f"  Saved background: {background_output_path}")

        # 합성
        print("\n[3/3] Compositing product...")
        final = self.overlay_product(background, product)

        # 저장
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            final_resized = final.resize(INSTAGRAM_SIZE, Image.LANCZOS)
            final_resized.convert("RGB").save(output_path, quality=95)
            print(f"\n  Resized to {INSTAGRAM_SIZE[0]}x{INSTAGRAM_SIZE[1]}")
            print(f"  Saved: {output_path}")

        print("\n" + "=" * 50)
        print("  Done!")
        print("=" * 50)

        return final
