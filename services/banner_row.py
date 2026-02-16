import io
import time
import os
import logging
import traceback
from PIL import Image, ImageFilter
from rembg import remove
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RATIO_TO_SIZE = {
    "1:1": (1024, 1024), "2:3": (848, 1264), "3:2": (1264, 848),
    "3:4": (896, 1200), "4:3": (1200, 896), "4:5": (928, 1152),
    "5:4": (1152, 928), "9:16": (768, 1376), "16:9": (1376, 768),
    "21:9": (1584, 672),
}

RATIO_CODE_MAP = {
    1: "1:1", "1": "1:1",
    2: "2:3", "2": "2:3",
    3: "3:2", "3": "3:2",
    4: "3:4", "4": "3:4",
    5: "4:3", "5": "4:3",
    6: "4:5", "6": "4:5",
    7: "5:4", "7": "5:4",
    8: "9:16", "8": "9:16",
    9: "16:9", "9": "16:9",
    10: "21:9", "10": "21:9",
}

class ProductPlacer:
    @staticmethod
    def place(bw, bh, ori, product_img):
        pw, ph = product_img.size
        aspect = pw / ph

        if ori == "landscape":
            # 가로: 오른쪽 중앙에 크게
            target_h = bh * 0.78
            w = int(target_h * aspect)
            h = int(target_h)
            if w > bw * 0.35:
                w = int(bw * 0.35)
                h = int(w / aspect)
            x = bw - w - int(bw * 0.05)
            y = (bh - h) // 2

        elif ori == "portrait":
            # 세로: 중앙에 크게
            target_w = bw * 0.65
            w = int(target_w)
            h = int(w / aspect)
            if h > bh * 0.40:
                h = int(bh * 0.40)
                w = int(h * aspect)
            x = (bw - w) // 2
            y = int(bh * 0.55) - (h // 2)

        else:  # square
            # 정사각: 중앙~약간 아래에 크게
            target_w = bw * 0.55
            w = int(target_w)
            h = int(w / aspect)
            if h > bh * 0.42:
                h = int(bh * 0.42)
                w = int(h * aspect)
            x = (bw - w) // 2
            y = int(bh * 0.55) - (h // 2)

        return {'pos': (x, y, w, h)}

    @staticmethod
    def get_layout_description(bw, bh, ori, placement):
        px, py, pw, ph = placement['pos']

        if ori == "landscape":
            return {
                'typo_position': "LEFT side of the banner",
                'deco_position': "CENTER area, between text and product. Also small elements on left/right edges",
                'product_desc': f"Product is on the RIGHT side ({pw}x{ph}px at position ({px},{py}))",
            }
        elif ori == "portrait":
            return {
                'typo_position': "TOP of the banner (upper 25%)",
                'deco_position': "LEFT and RIGHT SIDES of the product. Small objects flanking the product package",
                'product_desc': f"Product is at CENTER ({pw}x{ph}px at position ({px},{py}))",
            }
        else:  # square
            return {
                'typo_position': "TOP of the banner (upper 25%)",
                'deco_position': "LEFT and RIGHT SIDES of the product. Small objects flanking the product package",
                'product_desc': f"Product is at CENTER ({pw}x{ph}px at position ({px},{py}))",
            }

def create_guide_with_product(background, placement, product_img):
    guide = background.copy().convert("RGBA")
    px, py, pw, ph = placement['pos']
    prod_resized = product_img.resize((pw, ph), Image.LANCZOS)
    mask = prod_resized.split()[3]
    shadow = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    shadow.paste((0, 0, 0, 70), (0, 0), mask=mask)
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))
    guide.paste(shadow, (px + 8, py + 8), shadow)
    guide.paste(prod_resized, (px, py), prod_resized)
    return guide

class AdBannerGenerator:
    def __init__(self, api_key, model_name="gemini-2.0-flash-exp", ratio="1:1"):
        self.client = genai.Client(api_key=api_key)
        self.model = model_name
        
        # Resolve ratio
        if ratio in RATIO_CODE_MAP:
            self.ratio = RATIO_CODE_MAP[ratio]
        elif str(ratio) in RATIO_TO_SIZE:
            self.ratio = str(ratio)
        else:
            valid_keys = list(RATIO_CODE_MAP.keys()) + list(RATIO_TO_SIZE.keys())
            raise ValueError(f"Unsupported ratio: {ratio}. Valid values: {valid_keys}")

        self.bw, self.bh = RATIO_TO_SIZE[self.ratio]
        
        if self.bw > self.bh:
            self.ori = "landscape"
        elif self.bh > self.bw:
            self.ori = "portrait"
        else:
            self.ori = "square"

    def _retry(self, fn, retries=3, wait=5):
        for i in range(retries):
            try:
                return fn()
            except Exception as e:
                err = str(e)
                retryable = (
                    "429" in err or "RESOURCE_EXHAUSTED" in err
                    or "503" in err or "UNAVAILABLE" in err
                    or "overloaded" in err.lower()
                    or "INTERNAL" in err or "DEADLINE_EXCEEDED" in err
                )
                if retryable and i < retries - 1:
                    logger.warning(f"Retry {i+1}/{retries} due to error: {err}")
                    time.sleep(wait)
                else:
                    logger.error(f"Generate content failed after retries: {err}")
                    raise

    def remove_bg(self, path):
        orig = Image.open(path).convert("RGBA")
        try:
            prod = remove(orig)
        except Exception as e:
            logger.error(f"Background removal failed: {e}")
            logger.warning("Using original image as fallback")
            prod = orig
            
        bbox = prod.getbbox()
        if bbox:
            prod = prod.crop(bbox)
        return prod

    def analyze_product(self, image_path):
        orig = Image.open(image_path).convert("RGB")
        buf = io.BytesIO(); orig.save(buf, format="PNG")
        orig_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
        prompt = (
            "You are a creative director analyzing a product for an ad campaign.\n"
            "Return in English, one line each:\n\n"
            "PRODUCT_NAME:\nCATEGORY:\nKEY_INGREDIENTS_OR_FEATURES:\n"
            "DOMINANT_COLORS:\nMOOD:\n"
            "SUGGESTED_DECO: (small decorative object for sides of ad, e.g., shrimp, chili)\n"
            "BACKGROUND_SCENE:\n\n"
            "ONLY the 7 lines."
        )
        try:
            r = self.client.models.generate_content(
                model="gemini-2.0-flash-exp", # Using a fast model for analysis
                contents=[types.Content(role="user", parts=[
                    types.Part.from_text(text=prompt), orig_part,
                ])],
            )
            return r.text.strip()
        except Exception as e:
            logger.error(f"Product analysis failed: {e}")
            return "PRODUCT_NAME: snack\nMOOD: bold\nSUGGESTED_DECO: shrimp\nBACKGROUND_SCENE: dark kitchen"

    def generate_scene_concept(self, product_analysis, guideline):
        prompt = (
            "Create ONE background scene concept for a product ad.\n\n"
            f"PRODUCT:\n{product_analysis}\n\n"
        )
        if guideline and guideline.strip():
            prompt += f"GUIDELINE:\n{guideline}\n\n"
        prompt += (
            "Return EXACTLY:\n"
            "SCENE:\nGROUND:\nSKY_OR_CEILING:\nLIGHTING:\nATMOSPHERE:\n"
            "COLOR_PALETTE:\nAMBIENT_DETAILS:\nSIDE_DECO:\n\n"
            "Rules: ONE continuous photo, no text, no product."
        )
        try:
            r = self.client.models.generate_content(
                model="gemini-2.0-flash-exp", 
                contents=prompt,
            )
            return r.text.strip()
        except Exception as e:
            logger.error(f"Scene concept generation failed: {e}")
            return "SCENE: Dark dramatic kitchen\nSIDE_DECO: Flaming shrimp, chili peppers"

    def generate_pure_background(self, scene_concept, original_img_path, guideline=""):
        orig = Image.open(original_img_path).convert("RGB")
        buf = io.BytesIO(); orig.save(buf, format="PNG")
        orig_bytes = buf.getvalue()

        style_hint = ""
        if guideline and guideline.strip():
            for keyword in ["장소", "배경", "스타일", "느낌", "효과", "역동"]:
                for line in guideline.split("\n"):
                    if keyword in line:
                        style_hint += line.strip() + "\n"

        prompt = (
            f"Generate a {self.bw}x{self.bh}px ({self.ratio}) background photograph.\n\n"
            f"## SCENE:\n{scene_concept}\n\n"
        )
        if style_hint.strip():
            prompt += f"## ADDITIONAL STYLE DIRECTION:\n{style_hint}\n\n"
        prompt += (
            "## RULES:\n"
            "- ONE continuous photograph, uniform lighting\n"
            "- NO text, NO typography, NO letters\n"
            "- NO large foreground objects\n"
            "- NO product packages\n"
            "- NO humans or animals\n"
            "- NO color palette swatches or color bars\n"
            "- NO mood boards, collages, or multiple photos stitched together\n"
            "- NO borders, frames, or decorative edges\n"
            "- The entire canvas must be ONE seamless photograph\n"
            "- Small ambient details (particles, smoke) OK\n"
            "- Professional cinematic quality\n"
            "Generate now."
        )

        parts = [
            types.Part.from_text(text=prompt),
            types.Part.from_text(text="\n[COLOR REFERENCE ONLY] Do NOT draw this."),
            types.Part.from_bytes(data=orig_bytes, mime_type="image/png"),
        ]

        resp = self._retry(lambda: self.client.models.generate_content(
            model=self.model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
        ))

        for p in resp.candidates[0].content.parts:
            if p.inline_data and p.inline_data.mime_type.startswith("image/"):
                bg = Image.open(io.BytesIO(p.inline_data.data))
                return bg.resize((self.bw, self.bh), Image.LANCZOS)

        return Image.new("RGB", (self.bw, self.bh), (30, 30, 30))

    def add_typo_and_side_deco(self, guide_img, placement, typo_text, scene_concept, guideline=""):
        buf = io.BytesIO()
        guide_img.convert("RGB").save(buf, format="PNG")
        guide_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")

        px, py, pw, ph = placement['pos']
        layout = ProductPlacer.get_layout_description(
            self.bw, self.bh, self.ori, placement
        )

        side_deco = "small dramatic objects (shrimp, chili, flames)"
        if scene_concept:
            for line in scene_concept.split("\n"):
                if "SIDE_DECO" in line or "HERO_OBJECT" in line or "SUGGESTED_DECO" in line:
                    side_deco = line.split(":", 1)[-1].strip()
                    break

        max_deco_size = int(min(pw, ph) * 0.45)

        typo_style_lines = ""
        if guideline and guideline.strip():
            for keyword in ["글자", "폰트", "font", "타이포", "색", "color", "재질", "테두리", "stroke"]:
                for line in guideline.split("\n"):
                    if keyword in line.lower() or keyword in line:
                        typo_style_lines += line.strip() + "\n"
            seen = set()
            unique_lines = []
            for line in typo_style_lines.strip().split("\n"):
                if line and line not in seen:
                    seen.add(line)
                    unique_lines.append(line)
            typo_style_lines = "\n".join(unique_lines)

        if typo_style_lines.strip():
            typo_color_instruction = (
                f"- Typography style from user guideline (FOLLOW THIS STRICTLY):\n"
                f"  {typo_style_lines}\n"
            )
        else:
            typo_color_instruction = (
                "- Choose a color that CONTRASTS well with the background\n"
                "- If background is dark: use gold, white, or bright colors\n"
                "- If background is bright: use deep navy, dark blue with white outline\n"
            )

        prompt = (
            "I'm giving you a product ad banner with a product package already placed.\n"
            f"The product package is the MAIN HERO of this ad.\n"
            f"{layout['product_desc']}\n\n"

            "## YOUR TASKS:\n\n"

            f'### 1. Korean Typography: "{typo_text}"\n'
            f"- Position: {layout['typo_position']}\n"
            "- Large, bold, brush-stroke/display style\n"
            f"{typo_color_instruction}"
            "- EXACT characters, no translation\n"
            "- Must NOT cover the product package\n\n"

            f"### 2. Small Side Decorations: {side_deco}\n"
            f"- Position: {layout['deco_position']}\n"
            f"- Each decoration MUST be smaller than {max_deco_size}px in both width and height\n"
            "- They SUPPORT the product, they are NOT the star\n"
            "- Place them on the SIDES of the product, like flanking guards\n"
            "- They should NOT overlap or cover the product package\n"
            "- 2-4 small objects scattered around the sides\n"
            "- Can include: small flames, floating chilies, salt crystals, sparks\n\n"

            "## HIERARCHY:\n"
            "PRODUCT PACKAGE (biggest, center) > TYPOGRAPHY (big, top/left) > SIDE DECO (small, sides)\n\n"

            "## RULES:\n"
            "- Product package: DO NOT modify, move, or cover\n"
            "- Background: keep as-is\n"
            "- No other text besides Korean typography\n"
            "- No additional packages\n"
            "- No humans/animals\n\n"

            "Output the complete banner."
        )

        resp = self._retry(lambda: self.client.models.generate_content(
            model=self.model,
            contents=[types.Content(role="user", parts=[
                types.Part.from_text(text=prompt), guide_part,
            ])],
            config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
        ))

        for p in resp.candidates[0].content.parts:
            if p.inline_data and p.inline_data.mime_type.startswith("image/"):
                result = Image.open(io.BytesIO(p.inline_data.data))
                return result.resize((self.bw, self.bh), Image.LANCZOS)

        return guide_img.convert("RGB")

    def process(self, image_path, output_path, typo_text, guideline=""):
        # 1. Background Removal
        product = self.remove_bg(image_path)
        
        # 2. Analyze
        product_analysis = self.analyze_product(image_path)
        scene_concept = self.generate_scene_concept(product_analysis, guideline)

        # 3. Placement
        placement = ProductPlacer.place(self.bw, self.bh, self.ori, product)
        
        # 4. Generate Background
        background = self.generate_pure_background(scene_concept, image_path, guideline)
        
        # 5. Create Guide
        guide = create_guide_with_product(background, placement, product)
        
        # 6. Final Generation (Typo + Deco)
        final = self.add_typo_and_side_deco(
            guide, placement, typo_text, scene_concept, guideline
        )
        
        final.save(output_path)
        return output_path