import io
from PIL import Image, ImageFilter
from rembg import remove
from google import genai
from google.genai import types


class AdBannerGenerator:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.banner_width = 1680
        self.banner_height = 720

    def _remove_background(self, image_path):
        original = Image.open(image_path).convert("RGBA")
        try:
            product = remove(original)
        except:
            product = original
        bbox = product.getbbox()
        if bbox:
            product = product.crop(bbox)
        return product

    def _plan_background(self, product_img, headline, typo_text):
        prompt = f"""
        You are a Creative Director.
        - Overall Theme/Headline: "{headline}"
        - Exact Text to Display: "{typo_text}"
        Output ONLY the final prompt for the image generator.
        """
        response = self.client.models.generate_content(
            model="gemini-3-pro-preview", contents=[prompt, product_img]
        )
        return response.text

    def _draw_background(self, prompt):
        response = self.client.models.generate_content(
            model="gemini-3-pro-image-preview", contents=[prompt]
        )
        bg_image = None
        for part in response.parts:
            if part.inline_data:
                bg_image = Image.open(io.BytesIO(part.inline_data.data))
                break
        if bg_image:
            return bg_image.resize(
                (self.banner_width, self.banner_height), Image.LANCZOS
            )
        else:
            return Image.new(
                "RGB", (self.banner_width, self.banner_height), (200, 200, 200)
            )

    def _composite(self, background, product):
        mask = product.split()[3]
        shadow = Image.new("RGBA", product.size, (0, 0, 0, 0))
        shadow.paste((0, 0, 0, 80), (0, 0), mask=mask)
        shadow = shadow.filter(ImageFilter.GaussianBlur(20))

        scale = (self.banner_height * 0.75) / product.height
        new_w, new_h = int(product.width * scale), int(product.height * scale)
        product = product.resize((new_w, new_h), Image.LANCZOS)
        shadow = shadow.resize((new_w, new_h), Image.LANCZOS)

        bg_w, bg_h = background.size
        x = int(bg_w * 0.6) - (new_w // 2)
        y = (bg_h - new_h) // 2 + 50

        final = background.convert("RGBA")
        final.paste(shadow, (x + 10, y + 10), shadow)
        final.paste(product, (x, y), product)
        return final

    def process(self, image_path, headline, typo_text, output_path):
        product = self._remove_background(image_path)
        bg_prompt = self._plan_background(product, headline, typo_text)
        background = self._draw_background(bg_prompt)
        final_result = self._composite(background, product)
        final_result.save(output_path)
        return output_path
