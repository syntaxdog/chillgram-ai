"""
[Project] AI-Powered Box Packaging Mockup Generator
[Description]
이 스크립트는 전개도(Dieline) 이미지와 컨셉 이미지를 입력받아,
AI를 이용해 자연스러운 박스 패키지 디자인 시안을 생성합니다.

[Process]
1. 구조 분석 (Structure Analysis): 전개도의 패널 위치 및 크기를 인식 (OpenCV)
2. 배경 생성 (Background Generation): 자연스러운 종이 질감의 텍스처 생성 (GLOBAL BACKGROUND)
3. 패널 디자인 (Panel Design): 각 패널(메인, 사이드)에 독립적인 디자인 요소 생성 (ISOLATED ELEMENTS)
4. 반응형 배치 (Responsive Overlay): 생성된 요소를 각 패널 크기에 맞춰 리사이징 및 배치
5. 최종 합성 (Final Compositing): 마스킹 및 선명한 도면 선 복구
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
from google import genai
from google.genai import types
import io
import platform
import subprocess
import time

class DielineGenerator:
    def __init__(self, api_key: str = None):
        """
        [설정] API 키 및 모델 설정
        API_KEY가 주어지지 않으면 환경 변수나 기본값을 사용합니다.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            # Fallback to the hardcoded key from original file if they want
            self.api_key = ""
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_id = "gemini-3-pro-image-preview"

    def _generate_target_image(self, prompt, ref_img, w, h):
        """
        Google GenAI를 통해 이미지를 생성하는 함수. (3회 재시도 로직 추가)
        """
        if not isinstance(ref_img, list):
            ref_img = [ref_img]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=[prompt] + ref_img,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"], 
                        temperature=0.0,
                        safety_settings=[{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}]
                    )
                )
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data:
                            img = Image.open(io.BytesIO(part.inline_data.data)).convert("RGBA")
                            return img.resize((w, h), Image.Resampling.LANCZOS)
            except Exception as e:
                print(f">>> [API 에러] {e} (시도 {attempt+1}/{max_retries})")
                time.sleep(1)  # 잠시 대기 후 재시도
                
        return None

    def _make_square_canvas(self, img, fill_color=None):
        """
        Aspect Ratio Preservation (Smart Padding)
        """
        w, h = img.size
        size = max(w, h)
        
        # [Technique] Edge Padding (Textured Clamp) vs Solid Color
        if fill_color:
            new_img = Image.new("RGBA", (size, size), fill_color)
        else:
            new_img = Image.new("RGBA", (size, size), (0,0,0,0))
        
        # Center coordinates
        x = (size - w) // 2
        y = (size - h) // 2
        
        # Paste original centered
        new_img.paste(img, (x, y))
        
        if not fill_color:
            # 1. Fill Top Gap (if any)
            if y > 0:
                top_row = img.crop((0, 0, w, 1)) # Get top 1px line
                top_fill = top_row.resize((w, y), Image.Resampling.NEAREST)
                new_img.paste(top_fill, (x, 0))
                
            # 2. Fill Bottom Gap (if any)
            if y + h < size:
                bottom_row = img.crop((0, h-1, w, h)) # Get bottom 1px line
                bottom_fill = bottom_row.resize((w, size - (y+h)), Image.Resampling.NEAREST)
                new_img.paste(bottom_fill, (x, y+h))
                
            # 3. Fill Left/Right Gaps
            if x > 0:
                left_col = new_img.crop((x, 0, x+1, size))
                left_fill = left_col.resize((x, size), Image.Resampling.NEAREST)
                new_img.paste(left_fill, (0, 0))
                
            if x + w < size:
                right_col = new_img.crop((x+w-1, 0, x+w, size))
                right_fill = right_col.resize((size-(x+w), size), Image.Resampling.NEAREST)
                new_img.paste(right_fill, (x+w, 0))

        return new_img, (x, y, w, h)

    def _extract_and_resize(self, image, panel_info, canvas, max_scale=0.9):
        """
        Extracts content from a generated image (assuming white/solid background),
        resizes it responsively to fit the panel, and composites it onto the canvas.
        """
        if image is None: return

        # Convert to numpy for OpenCV processing
        img_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Check average brightness to decide thresholding (Dark text on Light BG vs Light text on Dark BG)
        mean_val = np.mean(gray)
        if mean_val > 200: # Mostly white background
             _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        else: # Mostly dark background
             _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

        coords = cv2.findNonZero(thresh)
        if coords is None:
            print(">>> [Responsive] Content not found in generated image.")
            return

        x, y, w, h = cv2.boundingRect(coords)
        content = image.crop((x, y, x+w, y+h)).convert("RGBA")
        
        # Simple White Removal (Make White Transparent)
        datas = content.getdata()
        new_data = []
        for item in datas:
            # Tolerance for white/near-white
            if item[0] > 230 and item[1] > 230 and item[2] > 230:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)
        content.putdata(new_data)

        # Responsive Resize
        panel_w = int(panel_info['w'])
        panel_h = int(panel_info['h'])
        
        target_w = int(panel_w * max_scale)
        target_h = int(panel_info['h'] * max_scale)
        
        cw, ch = content.size
        # Avoid division by zero
        if cw == 0 or ch == 0: return

        scale = min(target_w / cw, target_h / ch)
        new_w = int(cw * scale)
        new_h = int(ch * scale)
        
        if new_w <= 0 or new_h <= 0: return

        resized_content = content.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Center and Paste
        origin_x = int(panel_info['x'])
        origin_y = int(panel_info['y'])
        
        center_x = origin_x + (panel_w - new_w) // 2
        center_y = origin_y + (panel_h - new_h) // 2
        
        # Paste with alpha composite
        canvas.paste(resized_content, (center_x, center_y), resized_content)
        print(f">>> [Responsive] Placed element at ({center_x}, {center_y}) size ({new_w}x{new_h})")

    def generate(self, dieline_path: str, concept_path: str, output_path: str):
        """
        Main pipeline execution wrapper using provided paths.
        """
        # ==============================================================================
        # [1단계] 구조 분석 (Structure Analysis)
        # ==============================================================================
        print(">>> [1단계] 구조 분석 및 모든 패널 좌표 고정 중...")
        
        img_cv = cv2.imread(dieline_path)
        if img_cv is None: 
            print(f">>> [Error] Cannot load dieline: {dieline_path}")
            return
        
        h, w, _ = img_cv.shape
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        try:
            concept_img = Image.open(concept_path).convert("RGB")
        except Exception as e:
            print(f">>> [Error] Cannot load concept image: {concept_path}")
            return

        # 이진화 및 노이즈 제거
        _, thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((7, 7), np.uint8)
        walled = cv2.dilate(thresh, kernel, iterations=2)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cv2.bitwise_not(walled))
        
        rooms = []
        total_area = h * w
        
        for i in range(1, num_labels):
            rx, ry, rw, rh, area = stats[i]
            if not ((total_area * 0.01) < area < (total_area * 0.5)): continue
            if rw > (w * 0.9) and rh > (h * 0.9): continue
            ratio = max(rw, rh) / min(rw, rh)
            if ratio > 10: continue

            rooms.append({'x': rx, 'y': ry, 'w': rw, 'h': rh, 'area': area, 'center': centroids[i]})

        if len(rooms) < 2:
            print(">>> [오류] 패널을 충분히 찾지 못했습니다.")
            return

        # [패널 선별 로직]
        valid_sides = []
        target_main = max(rooms, key=lambda r: r['area'])
        print(f">>> [정보] 메인 패널 좌표: x={target_main['x']}, y={target_main['y']}, w={target_main['w']}, h={target_main['h']}")
        
        for r in rooms:
            if r == target_main: continue
            y_a, h_a = target_main['y'], target_main['h']
            y_b, h_b = r['y'], r['h']
            inter_y1 = max(y_a, y_b)
            inter_y2 = min(y_a + h_a, y_b + h_b)
            inter_h = max(0, inter_y2 - inter_y1)
            overlap_ratio = inter_h / min(h_a, h_b)
            if overlap_ratio > 0.5:
                valid_sides.append(r)

        side1 = None
        best_dist = float('inf')
        for r in valid_sides:
            if r['x'] > target_main['x']: # 오른쪽
                dist = r['x'] - (target_main['x'] + target_main['w'])
            else: # 왼쪽
                dist = target_main['x'] - (r['x'] + r['w'])
            if abs(dist) < best_dist:
                best_dist = abs(dist)
                side1 = r

        if side1:
            print(f">>> [성공] 사이드 패널 선택 완료: x={side1['x']}, w={side1['w']}")
        else:
            print(">>> [주의] 적절한 사이드 패널을 찾지 못했습니다. (메인 패널만 생성합니다)")

        top_panel = None
        best_y_dist = float('inf')
        for r in rooms:
            if r == target_main or r == side1: continue
            if r['y'] + r['h'] > target_main['y'] + 20: continue
            x_a, w_a = target_main['x'], target_main['w']
            x_b, w_b = r['x'], r['w']
            inter_x1 = max(x_a, x_b)
            inter_x2 = min(x_a + w_a, x_b + w_b)
            inter_w = max(0, inter_x2 - inter_x1)
            if inter_w / min(w_a, w_b) > 0.5:
                dist = target_main['y'] - (r['y'] + r['h'])
                if dist < best_y_dist:
                    best_y_dist = dist
                    top_panel = r
        
        if top_panel:
            print(f">>> [성공] 상단(Top) 패널 선택 완료: x={top_panel['x']}, y={top_panel['y']}")

        # ==============================================================================
        # [2단계] 이미지 생성 - GLOBAL BACKGROUND FIRST
        # ==============================================================================
        print(">>> [2단계] 글로벌 배경 텍스처 생성 중...")
        
        # [Fix] Concept Image Pre-processing: Smart Background Removal & Color Extraction
        # If the concept image is a product on white, we crop and MASK the background.
        # This prevents the AI from seeing "White Background" and making the texture white.
        concept_ref_for_bg = concept_img
        dominant_color_desc = "Main Brand Color"

        try:
            img_np = np.array(concept_img.convert("RGB"))
            h_img, w_img, _ = img_np.shape
            
            # --- 1. Smart Background Removal (Flood Fill) ---
            # Seed points: 4 corners. We assume the product is centered and corners are background.
            seeds = [(0, 0), (w_img-1, 0), (0, h_img-1), (w_img-1, h_img-1)]
            
            # Mask for floodFill (needs to be 2 pixels larger)
            flood_mask = np.zeros((h_img+2, w_img+2), np.uint8)
            # Flags: Fill with 255 (value), on the MASK only.
            flood_flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
            
            # Tolerance: Allow slight off-white (shadows, lighting)
            tolerance = (20, 20, 20)
            
            # Standard "White" reference to check if corners are actually background
            # If a corner is NOT bright (e.g. image fills the frame), we skip flood filling from there.
            
            filled_count = 0
            for seed in seeds:
                pixel = img_np[seed[1], seed[0]]
                if all(c > 200 for c in pixel): # Only treat as background if it's light color
                    cv2.floodFill(img_np, flood_mask, seed, (0,0,0), tolerance, tolerance, flood_flags)
                    filled_count += 1
            
            if filled_count > 0:
                # Crop mask back to image size
                # flood_mask is 255 where background is identified
                bg_mask = flood_mask[1:-1, 1:-1]
                
                # --- 2. Color Extraction (Ignore White & Background) ---
                # We want the color of the PRODUCT, not the text (often white) or background.
                
                # Content pixels = where bg_mask is 0
                content_locs = np.where(bg_mask == 0)
                if len(content_locs[0]) > 0:
                    content_pixels = img_np[content_locs]
                    
                    # Filter out internal white pixels (e.g. white text on chip bag)
                    # We want the BAG color.
                    valid_colors = []
                    for px in content_pixels:
                        # If pixel is NOT white/light-grey
                        if not (px[0] > 200 and px[1] > 200 and px[2] > 200):
                            valid_colors.append(px)
                            
                    if len(valid_colors) > 0:
                        avg_color = np.mean(valid_colors, axis=0).astype(int)
                        dominat_rgb = tuple(avg_color)
                        dominant_color_desc = f"RGB{dominat_rgb} (Dark/Vibrant Brand Color)"
                        print(f">>> [Color Analysis] Detected Dominant Brand Color: {dominant_color_desc}")
                    else:
                        print(">>> [Color Analysis] Only white content found (e.g. White bag?). Using default.")
                        dominant_color_desc = "White/Silver"
                
                # --- 3. Create Masked Reference Image ---
                # Make background transparent for the AI reference
                concept_rgba = concept_img.convert("RGBA")
                datas = concept_rgba.getdata()
                new_data = []
                
                # We can use the numpy mask to do this faster, but for safety with PIL logic:
                # Re-constructing byte array
                bg_mask_flat = bg_mask.flatten() # 0 is content, 255 is bg
                
                # Create a new image with Transparency
                concept_arr_rgba = np.array(concept_rgba)
                # Set Alpha to 0 where bg_mask is 255
                concept_arr_rgba[:, :, 3] = np.where(bg_mask == 255, 0, 255)
                
                concept_ref_for_bg = Image.fromarray(concept_arr_rgba)
                
                # Optional: Crop to content bounds
                bbox = concept_ref_for_bg.getbbox()
                if bbox:
                    concept_ref_for_bg = concept_ref_for_bg.crop(bbox)
                    print(f">>> [Pre-process] Cropped to content size: {bbox}")
                
            else:
                print(">>> [Pre-process] No white background detected at corners. Using full image.")

        except Exception as e:
            print(f">>> [Color Analysis Error] {e}. Using full image.")
            concept_ref_for_bg = concept_img
        
        bg_prompt = f"""
        Create a **High-End Premium Packaging Material Texture**.
        
        1. **Color Analysis (CRITICAL)**:
           - The input image has had its background removed.
           - **THE DETECTED MAIN COLOR IS: {dominant_color_desc}**
           - **YOU MUST USE THIS COLOR ({dominant_color_desc}) for the texture.**
           - **DO NOT MAKE IT WHITE.** Even if the input looks light, force the detected color.
           
        2. **Material Finish**:
           - **Matte Coated Paper**: Smooth, slight sheen.
           - **Grain/Noise**: Subtle texture.
           - **FLAT SHEET**: No folds, no creases, no shadows.
           - **Solid Color with Texture**: Uniform color across the whole sheet.
        """
        bg_sheet = self._generate_target_image(bg_prompt, concept_ref_for_bg, w, h)
        if bg_sheet is None:
            print(">>> [Warning] Background Generation FAILED. Using fallback color (White).") 
            bg_sheet = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        else:
            print(">>> [Success] Background Generated.")
            
        final_canvas = bg_sheet.convert("RGBA")

        # ==============================================================================
        # [3단계] 패널별 디자인 생성 (Isolated Elements) & Overlay
        # ==============================================================================
        
        # 3-1. 사이드 패널 (Vertical Text Column)
        if side1:
            print(">>> [3단계] 사이드 패널: 정보 디자인 추출 및 오버레이...")
            
            # Use a vertical aspect ratio canvas for generation
            side_gen_w, side_gen_h = 512, 1024
            side_bg_white = Image.new("RGBA", (side_gen_w, side_gen_h), (255, 255, 255, 255))

            side_prompt = f"""
            **Role**: Expert Packaging Typographer.
            **Task**: Create a **VERTICAL Typography Layout** for a side panel.
            
            **Input**:
            - **Image 1**: Product Concept (Source of fonts/colors).
            - **Image 2**: White Canvas.

            **Requirements**:
            1. **Content**: 
               - **MUST INCLUDE**: The **Brand Logo / Trademark Graphic**.
               - **MUST INCLUDE**: The **Brand Name Text**.
            2. **Layout**: Vertical arrangement. Floating text.
            3. **Style**: Matching the concept image font.
            
            **CRITICAL CONSTRAINTS**:
            - **NO NUTRITION FACTS TABLE.**
            - **NO INGREDIENTS LIST.**
            - **NO SLOGANS. NO WEBSITE URL. NO SELLING POINTS.**
            - **NO BARCODE.**
            - **NO Background Texture**: Keep background Pure White.
            """
            
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=[side_prompt, concept_img, side_bg_white],
                    config=types.GenerateContentConfig(response_modalities=["IMAGE"], temperature=0.0)
                )
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data:
                            img_data = Image.open(io.BytesIO(part.inline_data.data)).convert("RGBA")
                            self._extract_and_resize(img_data, side1, final_canvas, max_scale=0.85)
            except Exception as e:
                print(f">>> [Side Panel Error] {e}")

        # 3-2. 상단 패널 (Logo Only)
        if top_panel:
            print(">>> [3.5단계] 상단 패널: 로고 추출 및 오버레이...")
            
            top_gen_w, top_gen_h = 1024, 512
            top_bg_white = Image.new("RGBA", (top_gen_w, top_gen_h), (255, 255, 255, 255))
            
            top_prompt = f"""
            **Role**: Brand Designer.
            **Task**: Extract **Brand Logo & Name** for the Top Lid.
            
            **Input**:
            - **Image 1**: Product Concept.
            - **Image 2**: White Canvas.
            
            **Requirements**:
            1. **Output**: The Logo on **PURE WHITE** background.
            2. **Content**: 
               - **MUST INCLUDE**: The **Brand Logo/Trademark Graphic**.
               - **MUST INCLUDE**: The **Brand Name Text**.
            3. **Style**: Flat Vector.
            4. **Orientation**: Horizontal.

            **CRITICAL CONSTRAINTS**:
            - **NO NUTRITION FACTS.**
            - **NO SLOGANS. NO LONG TEXT BLOCKS.**
            """
            
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=[top_prompt, concept_img, top_bg_white],
                    config=types.GenerateContentConfig(response_modalities=["IMAGE"], temperature=0.0)
                )
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data:
                            img_data = Image.open(io.BytesIO(part.inline_data.data)).convert("RGBA")
                            self._extract_and_resize(img_data, top_panel, final_canvas, max_scale=0.7)
            except Exception as e:
                print(f">>> [Top Panel Error] {e}")

        # 3-3. 메인 패널 (Main Graphic - Chroma Keyed)
        print(">>> [4단계] 메인 패널: 디자인 요소 추출 및 오버레이...")
        
        chroma_size = 1024
        chroma_sq_ref = Image.new("RGBA", (chroma_size, chroma_size), (255, 0, 255, 255))

        main_prompt = f"""
        **Role**: Senior Packaging Illustrator.
        **Task**: Extract **Main GRAPHICS & LOGO** for the front panel.

        **Input**:
        - **Image 1**: Product Bag (Source).
        - **Image 2**: Magenta Green Screen (Canvas).

        **Execution Steps**:
        1. **Extraction**: 
           - **MUST INCLUDE**: The Main Brand Logo / Trademark (Center/Top).
           - **MUST INCLUDE**: Main Illustration / Product Image.
           - **MUST INCLUDE**: Selling Points / Flavor Name.
        2. **Composition on Magenta**:
           - Place these elements on the **Magenta Background**.
           - **Scale**: Make them LARGE and FILL the canvas.
           - **Centering**: Perfectly center the main logo.
        
        **Style Guide**:
        - **Outcome**: A flat, digital **Print File**.
        - **Shadows**: Drop shadows allowed ONLY if part of text.
        - **NO** 3D Bag mesh.

        **CRITICAL CONSTRAINTS**:
        - **NO NUTRITION FACTS.**
        - **NO INGREDIENTS LIST.**
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[main_prompt, concept_img, chroma_sq_ref],
                config=types.GenerateContentConfig(response_modalities=["IMAGE"], temperature=0.0)
            )
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        raw_art = Image.open(io.BytesIO(part.inline_data.data)).convert("RGBA")
                        
                        # Apply Chroma Key Logic (Refined)
                        frame = np.array(raw_art.convert("RGB"))
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        
                        lower_magenta = np.array([125, 30, 30])
                        upper_magenta = np.array([175, 255, 255])
                        
                        mask_inv = cv2.inRange(hsv, lower_magenta, upper_magenta)
                        mask = cv2.bitwise_not(mask_inv)
                        
                        # Refinement
                        kernel = np.ones((3,3), np.uint8)
                        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                        
                        # Fallback Logic
                        bg_ratio = np.count_nonzero(mask_inv) / mask_inv.size
                        if bg_ratio < 0.05:
                            print(">>> [Chroma Key] Fallback to Corner Sampling")
                            h_art, w_art, _ = frame.shape
                            corners = [frame[0,0], frame[0, w_art-1], frame[h_art-1, 0], frame[h_art-1, w_art-1]]
                            bg_mean = np.mean(corners, axis=0)
                            diff = frame.astype(float) - bg_mean
                            dist = np.sqrt(np.sum(diff**2, axis=2))
                            mask = (dist > 60).astype(np.uint8) * 255
                        
                        mask_pil = Image.fromarray(mask.astype(np.uint8), mode="L")
                        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=1))
                        
                        bbox = mask_pil.getbbox()
                        if bbox:
                            cropped_art = raw_art.crop(bbox)
                            cropped_mask = mask_pil.crop(bbox)
                            cropped_art.putalpha(cropped_mask)
                            
                            # Responsive Resize for Main Panel
                            panel_w = int(target_main['w'])
                            panel_h = int(target_main['h'])
                            target_w = int(panel_w * 0.95)
                            target_h = int(panel_h * 0.95)
                            
                            cw, ch = cropped_art.size
                            scale = min(target_w / cw, target_h / ch)
                            new_w = int(cw * scale)
                            new_h = int(ch * scale)
                            
                            resized_art = cropped_art.resize((new_w, new_h), Image.Resampling.LANCZOS)
                            
                            origin_x = int(target_main['x'])
                            origin_y = int(target_main['y'])
                            center_x = origin_x + (panel_w - new_w) // 2
                            center_y = origin_y + (panel_h - new_h) // 2
                            
                            final_canvas.paste(resized_art, (center_x, center_y), resized_art)
                        else:
                            # Fallback
                            raw_art = raw_art.resize((int(target_main['w']), int(target_main['h'])))
                            final_canvas.paste(raw_art, (int(target_main['x']), int(target_main['y'])))

        except Exception as e:
            print(f">>> [Main Panel Error] {e}")

        # ==============================================================================
        # [5단계] 최종 합성 (Final Compositing)
        # ==============================================================================
        print(">>> [5단계] 도면 마스킹 및 선명도 복원 (Crisp Dieline)...")
        
        gray_dieline = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_dieline, 235, 255, cv2.THRESH_BINARY_INV)
        
        # Kernel for Mask Generation
        kernel_mask = np.ones((5, 5), np.uint8)
        walled = cv2.dilate(thresh, kernel_mask, iterations=1)
        
        num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(cv2.bitwise_not(walled))
        
        bg_label = labels_map[0, 0]
        mask = np.zeros(labels_map.shape, dtype=np.uint8)
        
        for i in range(1, num_labels):
            if i == bg_label: continue
            if stats[i, cv2.CC_STAT_AREA] < (total_area * 0.0001):
                continue
            mask[labels_map == i] = 255
        
        # [Important] Bleed Dilation
        dilation_kernel = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask, dilation_kernel, iterations=2)
        
        mask_pil = Image.fromarray(mask).convert("L")
        white_bg = Image.new("RGB", (w, h), (255, 255, 255))
        result_img = Image.composite(final_canvas.convert("RGB"), white_bg, mask_pil)
        
        # [Dieline Enhancement]
        dieline_orig = Image.open(dieline_path).convert("L")
        enhancer = ImageEnhance.Contrast(dieline_orig)
        dieline_enhanced = enhancer.enhance(3.0)
        dieline_np = np.array(dieline_orig)
        _, dieline_sharp = cv2.threshold(dieline_np, 200, 255, cv2.THRESH_BINARY)
        dieline_final_pil = Image.fromarray(dieline_sharp).convert("RGB")
        
        # Multiply
        final_output = ImageChops.multiply(result_img, dieline_final_pil)

        # Output Path logic
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        final_output.save(output_path)

        print(f"\n>>> [완료] 최종 결과물이 저장되었습니다: {output_path}")

if __name__ == "__main__":

    import sys
    
    # Default values
    dieline_input = "dieline_input.png"
    concept_input = "concept_input.png"
    output_result = "dieline_result.png"
    
    # Parse arguments
    if len(sys.argv) >= 2:
        dieline_input = sys.argv[1]
    if len(sys.argv) >= 3:
        concept_input = sys.argv[2]
    if len(sys.argv) >= 4:
        output_result = sys.argv[3]
        
    if os.path.exists(dieline_input) and os.path.exists(concept_input):
        print(f"Starting Dieline Generator...")
        print(f" - Dieline: {dieline_input}")
        print(f" - Concept: {concept_input}")
        print(f" - Output:  {output_result}")

        # API KEY logic for standalone run
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
             print("Warning: GEMINI_API_KEY not found in env.")
        
        try:
            generator = DielineGenerator(api_key=api_key)
            generator.generate(dieline_input, concept_input, output_result)
            
            if platform.system() == "Darwin" and os.path.exists(output_result): 
                subprocess.run(["open", output_result])
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Usage: python services/dieline_generate.py [dieline_path] [concept_path] [output_path]")
        print(f"Files not found:")
        if not os.path.exists(dieline_input): print(f" - Missing: {dieline_input}")
        if not os.path.exists(concept_input): print(f" - Missing: {concept_input}")