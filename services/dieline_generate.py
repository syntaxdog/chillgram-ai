"""
[Project] AI-Powered Box Packaging Mockup Generator
[Description]
이 스크립트는 전개도(Dieline) 이미지와 컨셉 이미지를 입력받아,
AI를 이용해 자연스러운 박스 패키지 디자인 시안을 생성합니다.

[Process]
1. 구조 분석 (Structure Analysis): 전개도의 패널 위치 및 크기를 인식 (OpenCV)
2. 배경 생성 (Background Generation): 자연스러운 종이 질감의 텍스처 생성 (Google GenAI)
3. 패널 디자인 (Panel Design): 각 패널(메인, 사이드)에 평면 2D 아트워크 생성 및 합성
4. 최종 합성 (Final Compositing): 마스킹 및 선명한 도면 선 복구
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from google import genai
from google.genai import types
import io
import platform
import subprocess
import time

# ==============================================================================
# [설정] API 키 및 모델 설정
# ==============================================================================
API_KEY = "AIzaSyDpN5bpDXca4KZkzUwLQUYY6ysDY7Q5FnE"
MODEL_ID = "gemini-3-pro-image-preview"

client = genai.Client(api_key=API_KEY)

def generate_target_image(prompt, ref_img, w, h):
    """
    Google GenAI를 통해 이미지를 생성하는 함수. (3회 재시도 로직 추가)
    """
    if not isinstance(ref_img, list):
        ref_img = [ref_img]
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[prompt] + ref_img,
                config=types.GenerateContentConfig(response_modalities=["IMAGE"], temperature=0.0)
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

def paste_nukki(canvas, art, pos):
    """
    흰색 배경의 아트워크에서 포그라운드 요소만 '누끼'를 따서 캔버스에 합성합니다.
    """
    if art is None: return
    # 흰색 부분을 찾아서 마스크 생성 (Threshold: 250)
    # L(Grayscale) 채널에서 어두운 부분(요소)을 255로, 밝은 부분(배경)을 0으로 변환
    grayscale = art.convert("L")
    mask = grayscale.point(lambda p: 255 if p < 250 else 0).convert("L")
    canvas.paste(art, pos, mask)

def run_final_natural_pipeline(dieline_path, concept_path):
    # ==============================================================================
    # [1단계] 구조 분석 (Structure Analysis)
    # - 전개도 이미지의 윤곽선을 분석하여 각 패널(면)의 좌표를 추출합니다.
    # ==============================================================================
    print(">>> [1단계] 구조 분석 및 모든 패널 좌표 고정 중...")
    
    img_cv = cv2.imread(dieline_path)
    if img_cv is None: return
    
    h, w, _ = img_cv.shape
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    concept_img = Image.open(concept_path).convert("RGB")

    # 이진화 및 노이즈 제거
    _, thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((9, 9), np.uint8)
    walled = cv2.dilate(thresh, kernel, iterations=2)
    
    # 연결된 컴포넌트(패널 후보) 추출
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cv2.bitwise_not(walled))
    
    rooms = []
    total_area = h * w
    
    for i in range(1, num_labels):
        rx, ry, rw, rh, area = stats[i]
        # [설정] 너무 작거나 큰 영역은 제외 (전체 면적의 1% ~ 50%)
        if not ((total_area * 0.01) < area < (total_area * 0.5)):
            continue

        # [Fix] Bounding Box Limit: Ignore "Frame" panels that cover the whole image
        # If a panel is >90% of width AND >90% of height, it's the background container.
        if rw > (w * 0.9) and rh > (h * 0.9):
            continue

        rooms.append({'x': rx, 'y': ry, 'w': rw, 'h': rh, 'area': area, 'center': centroids[i]})

    if len(rooms) < 2:
        print(">>> [오류] 패널을 충분히 찾지 못했습니다.")
        return

    # --------------------------------------------------------------------------
    # [패널 선별 로직]
    # 1. 메인 패널: 가장 넓은 면적을 가진 영역
    # 2. 사이드 패널: 메인 패널과 높이(Y축)가 비슷하게 겹치는 옆면 (뚜껑 제외)
    # --------------------------------------------------------------------------
    valid_sides = []
    print(f">>> [처리] 총 {len(rooms)}개의 패널 후보 발견. 정밀 정렬 확인 중...")

    # 메인 패널 식별 (최대 면적)
    target_main = max(rooms, key=lambda r: r['area'])
    print(f">>> [정보] 메인 패널 좌표: x={target_main['x']}, y={target_main['y']}, w={target_main['w']}, h={target_main['h']}")
    
    for r in rooms:
        if r == target_main: continue
        
        # 수직 오버랩(Vertical Overlap) 계산
        y_a, h_a = target_main['y'], target_main['h']
        y_b, h_b = r['y'], r['h']
        
        inter_y1 = max(y_a, y_b)
        inter_y2 = min(y_a + h_a, y_b + h_b)
        inter_h = max(0, inter_y2 - inter_y1)
        
        # 겹치는 비율이 50% 이상인 경우에만 '옆면'으로 간주 (상단 뚜껑 등 제외)
        overlap_ratio = inter_h / min(h_a, h_b)
        
        if overlap_ratio > 0.5:
            valid_sides.append(r)
            # print(f">>> [DEBUG] 유효한 사이드 패널 후보: x={r['x']}, overlap={overlap_ratio:.2f}")
        else:
            # print(f">>> [DEBUG] 제외된 패널 (정렬 안됨): x={r['x']}, overlap={overlap_ratio:.2f}")
            pass

    # X축 거리가 가장 가까운 사이드 패널 선택 (가장 인접한 우측/좌측 면)
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

    # ==============================================================================
    # [Top Panel Detection] 상단 패널(뚜껑) 찾기
    # - 메인 패널 바로 위에 위치하고(y값이 작음), 가로(X축) 정렬이 맞는 패널
    # ==============================================================================
    top_panel = None
    best_y_dist = float('inf')
    
    for r in rooms:
        if r == target_main or r == side1: continue
        
        # 1. 위치 검사: 메인 패널보다 위에 있어야 함 (y + h <= main.y + 여유)
        if r['y'] + r['h'] > target_main['y'] + 20: 
            # 메인보다 아래에 있거나 너무 많이 겹치면 패스
            continue
            
        # 2. 가로 정렬 검사: 메인 패널과 X축으로 겹치는 구간 확인
        x_a, w_a = target_main['x'], target_main['w']
        x_b, w_b = r['x'], r['w']
        
        inter_x1 = max(x_a, x_b)
        inter_x2 = min(x_a + w_a, x_b + w_b)
        inter_w = max(0, inter_x2 - inter_x1)
        
        # 가로폭의 50% 이상 겹치면 '같은 열'에 있는 것으로 간주
        if inter_w / min(w_a, w_b) > 0.5:
            # 3. 가장 가까운 패널 선택 (메인 패널 바로 위 = 거리가 가장 짧은 것)
            dist = target_main['y'] - (r['y'] + r['h'])
            if dist < best_y_dist:
                best_y_dist = dist
                top_panel = r

    if top_panel:
        print(f">>> [성공] 상단(Top) 패널 선택 완료: x={top_panel['x']}, y={top_panel['y']}")

    # ==============================================================================
    # [2단계] 이미지 생성 (Generative AI)
    # - 패널별로 최적화된 프롬프트를 사용하여 2D 텍스처를 생성합니다.
    # ==============================================================================
    print(">>> [2단계] 자연스러운 배경과 그래픽 생성 시작...")
    
    # 1. 공통 배경 (Signature Color & Texture)
    bg_prompt = """
    A seamless, high-quality Coated Cardboard texture background.
    Color: 
    - STRICTLY MAINTAIN the dominant brand color from the provided concept image.
    - If the concept is orange, be orange. If green, be green.

    Texture & Pattern:
    - Analyzes the concept image for subtle patterns (e.g., paper grain, noise, faint geometric shapes).
    - Apply a rich, professional texture that makes the package look premium.
    - Examples: Matte coating, subtle paper fiber, soft radial gradient, or faint repetitive brand motifs if present.
    
    Style: Clean, Commercial Packaging.
    """
    bg_sheet = generate_target_image(bg_prompt, concept_img, w, h)
    if bg_sheet is None:
        print(">>> [Warning] Background Generation FAILED. Using fallback color (White).") 
        bg_sheet = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    else:
        print(">>> [Success] Background Generated.")
        
    final_canvas = bg_sheet.convert("RGBA")

    # ==============================================================================
    # [Helper] Aspect Ratio Preservation (Smart Padding)
    # ==============================================================================
    def make_square_canvas(img, fill_color=None):
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

    # 2. 사이드 패널 디자인 (Typography on Texture)
    if side1:
        print(">>> [3단계] 사이드 패널: 텍스트 정보 디자인 생성 중...")
        s_bleed = 3
        # [Fix] Calc safe dimension based on actual panel size
        sx, sy = max(0, int(side1['x']) - s_bleed), max(0, int(side1['y']) - s_bleed)
        sw, sh = min(w, int(side1['w']) + s_bleed*2), min(h, int(side1['h']) + s_bleed*2)
        
        side_bg_ref = final_canvas.crop((sx, sy, sx+sw, sy+sh))
        
        # [Fix] Apply Square Canvas to Side Panel (Vertical Strip in Square)
        side_bg_sq, (pad_x, pad_y, target_w, target_h) = make_square_canvas(side_bg_ref, fill_color=(0, 0, 0, 255))

        side_prompt = f"""
        A clean, commercial 2D vector typography overlay for a side panel.
        Input Reference Context:
        - Image 1: Product Concept (Style Source).
        - Image 2: Background Texture (Vertical Strip in Center with BLACK Padding).

        Action:
        1. **Look at Image 2**: It is a vertical textured strip centered in a black square.
        2. **Work ONLY INSIDE this Vertical Textured Strip**. Ignore the black padding areas.
        3. **SEAMLESS INTEGRATION**: 
           - The background texture from Image 2 MUST be preserved. 
           - Do NOT erase the texture with white color.
           - We will PASTE your result directly. It must look continuous with the original texture.
        4. **REPLICATION (CRITICAL)**: 
           - Identify the Brand Name ("새우깡") and ANY other text/logos on the side of the box in Image 1.
           - Re-create them EXACTLY as they appear, preserving fonts, colors, and layout.
           - Orient the main text **VERTICALLY (Top-to-Bottom)**.
        5. **LAYOUT**:
           - **Brand Name ONLY**: Center (Vertical, Large).
           - **NO Manufacturer Logo**: Do not include "농심" or other logos on the side.
           - **NO Nutrition/Info**: Do not include any small text or tables. 
           - **CLEAN TEXTURE**: The rest of the strip should be the clean background texture.

        Style: Identical to the Product Concept. Flat Vector.
        **CRITICAL NEGATIVE CONSTRAINTS:** 
        - **NO WHITE BORDERS**: The texture must bleed to the edge of the strip.
        - **NO EXTRA ELEMENTS**: Absolutely NO other text, numbers, or logos. ONLY the Brand Name.
        - **NO FRAMES**: Do not draw a box around the content.
        - **NO BLEED**: Content must stay strictly inside the strip.
        """
        
        # [Fix] Manual Generation Block for Side Panel (Square -> Crop)
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[side_prompt, concept_img, side_bg_sq],
                config=types.GenerateContentConfig(response_modalities=["IMAGE"], temperature=0.0)
            )
            side_art = None
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        # 1. Load Square Result
                        sq_result = Image.open(io.BytesIO(part.inline_data.data)).convert("RGBA")
                        # 2. Resize to match Input Square
                        sq_w, sq_h = side_bg_sq.size
                        sq_result = sq_result.resize((sq_w, sq_h), Image.Resampling.LANCZOS)
                        # 3. Crop back to original Aspect Ratio (The center strip)
                        side_art = sq_result.crop((pad_x, pad_y, pad_x + target_w, pad_y + target_h))
                        # 4. Resize to target pixels
                        side_art = side_art.resize((sw, sh), Image.Resampling.LANCZOS)
            
            if side_art:
                # [Fix] Direct Paste for Seamless Texture (No Nukki)
                final_canvas.paste(side_art, (sx, sy))
        except Exception as e:
            print(f">>> [Side Panel Error] {e}")

    # 3. 상단 패널 디자인 (Top Panel - Logo Overlay)


    # 4. 메인 패널 디자인 (Rich, Flat 2D Design)
    print(">>> [4단계] 메인 패널: 프리미엄 박스 패키지 디자인 생성 중...")

    # [Top Panel - Logo Overlay] 메인 패널 생성 전에 상단 패널 디자인 적용
    if top_panel:
        print(">>> [3.5단계] 상단 패널: 로고 오버레이 생성 중...")
        t_bleed = 7
        tx, ty = max(0, int(top_panel['x']) - t_bleed), max(0, int(top_panel['y']) - t_bleed)
        tw, th = min(w, int(top_panel['w']) + t_bleed*2), min(h, int(top_panel['h']) + t_bleed*2)
        
        top_bg_ref = final_canvas.crop((tx, ty, tx+tw, ty+th))

        # [Technique] Aspect Ratio Preservation (Smart Padding)
        # Top Panel is wide. AI output is square. Resizing Square->Wide squashes height (stretches text width).
        # Solution: Pad the wide reference into a Square canvas -> Generate -> Crop back.
        # Uses global make_square_canvas helper.

        top_bg_ref_sq, (pad_x, pad_y, target_w, target_h) = make_square_canvas(top_bg_ref, fill_color=(0, 0, 0, 255))
        
        top_prompt = f"""
        A clean, commercial 2D vector logo layout for a box TOP panel.
        Input Reference Context:
        - Image 1: Product Concept (Style Source).
        - Image 2: Background Texture (Wide Textured Strip in Center with BLACK Padding).
        
        Action:
        1. Look at Image 2. It is a wide textured strip in the middle of a black square.
        2. Work **ONLY INSIDE THIS TEXTURED STRIP**. Ignore the black padding.
        3. **SEAMLESS INTEGRATION**:
           - The background texture MUST extend to the very edges of the strip.
           - NO WHITE GAPS at top/bottom.
           - We will PASTE your result directly. It must contain the texture.
        4. **STYLE REPLICATION (CRITICAL)**:
           - Analyze the Brand Name "새우깡" in Image 1.
           - **PRESERVE** the exact font weight, stroke style, and artistic calligraphy.
           - **DO NOT** change the font to a generic one. It must look like the original brand logo.
        5. **ORIENTATION**: 
           - Re-orient the logo to be **HORIZONTAL (Left-to-Right)** for the top lid.
        6. **LAYOUT**:
           - Center the horizontal logo.
           - Include the manufacturer logo ("농심") small near the brand name.
        
        Style: EXACT REPLICA of the Brand Logo on Texture.
        **CRITICAL NEGATIVE CONSTRAINTS:**
        - **NO VERTICAL TEXT**: Top panel text must be horizontal.
        - **NO WHITE BORDERS**: Do not draw a frame or border.
        - **NO EXTRA ELEMENTS**: No background patterns, just the logo on the texture.
        """
        
        # Generate Square Output
        # Note: We pass the SQUARE reference. The AI outputs a SQUARE image.
        # We manually handle the result instead of using generate_target_image's auto-resize
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[top_prompt, concept_img, top_bg_ref_sq],
                config=types.GenerateContentConfig(response_modalities=["IMAGE"], temperature=0.0)
            )
            top_art = None
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        # 1. Load Square Result
                        sq_result = Image.open(io.BytesIO(part.inline_data.data)).convert("RGBA")
                        # 2. Resize to match the Input Square Dimension (just in case model output differs slightly)
                        sq_w, sq_h = top_bg_ref_sq.size
                        sq_result = sq_result.resize((sq_w, sq_h), Image.Resampling.LANCZOS)
                        # 3. Crop back to original Aspect Ratio (The center strip)
                        top_art = sq_result.crop((pad_x, pad_y, pad_x + target_w, pad_y + target_h))
                        # 4. Resize to exact target pixels (fine-tuning)
                        top_art = top_art.resize((tw, th), Image.Resampling.LANCZOS)
            
            if top_art:
                # [Fix] Direct Paste for Seamless Texture
                final_canvas.paste(top_art, (tx, ty))
        except Exception as e:
            print(f">>> [Top Panel Error] {e}")
    print(">>> [4단계] 메인 패널: 프리미엄 박스 패키지 디자인 생성 중...")
    m_bleed = 22
    mx, my = max(0, int(target_main['x']) - m_bleed), max(0, int(target_main['y']) - m_bleed)
    mw, mh = min(w, int(target_main['w']) + m_bleed*2), min(h, int(target_main['h']) + m_bleed*2)
    
    # 배경 레퍼런스 캡처
    bg_ref = final_canvas.crop((mx, my, mx+mw, my+mh))
    
    # [Fix] Apply Square Canvas to Main Panel (Responsive Design)
    # Important: We want AI to generate on MAGENTA background (Chroma Key), so we feed it a MAGENTA canvas
    # But we still need the size reference.
    bg_ref_sq, (pad_x, pad_y, target_w, target_h) = make_square_canvas(bg_ref, fill_color=(0, 0, 0, 255))
    
    # Create a Pure Magenta Square Canvas for the AI to draw on
    # Magenta (255, 0, 255) is a standard chroma key color unlikely to be in food packaging
    chroma_color = (255, 0, 255, 255)
    chroma_sq_ref = Image.new("RGBA", bg_ref_sq.size, chroma_color)

    main_prompt = f"""
    You are a professional packaging designer.
    Action: Deconstruct the product packaging image and extract the **FLAT PRINT DESIGN FILE**.
    
    Input:
    - Image 1: A photograph of a product bag (e.g., Shrimp Crackers).
    - Image 2: A Magenta Canvas (Target Output Area).

    Task:
    1. **IGNORE THE BAG OBJECT**: Do NOT draw the crinkled bag, the plastic, the shading, or the 3D shape.
    2. **EXTRACT GRAPHICS ONLY**: Extract the Logo, Product Illustration (Shrimp), and Main Text as **FLAT 2D VECTORS**.
    3. **RE-ASSEMBLE**: Arrange these graphics on the Magenta Canvas to match the original layout.
    4. **FULL BLEED LAYOUT**: Make the design LARGE. Fill the canvas as much as possible. Minimal margins.
    
    Target Output:
    - **Background**: MAGENTA (#FF00FF).
    - **Content**: Clean, high-resolution vector graphics of the brand and product.
    - **Style**: Adobe Illustrator Vector Art. NOT a photo of a bag.
    
    **CRITICAL NEGATIVE CONSTRAINTS:**
    - **NO BAG WRINKLES**: The result must look like a digital print file, not a photo.
    - **NO SHADING/GLOSSY REFLECTIONS**: Flat colors only.
    - **NO BACKGROUND SCENE**: Just the graphics on magenta.
    """
    
    # Helper to generate square and crop back
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[main_prompt, concept_img, chroma_sq_ref],
            config=types.GenerateContentConfig(response_modalities=["IMAGE"], temperature=0.0)
        )
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    sq_result = Image.open(io.BytesIO(part.inline_data.data)).convert("RGBA")
                    sq_w, sq_h = bg_ref_sq.size
                    sq_result = sq_result.resize((sq_w, sq_h), Image.Resampling.LANCZOS)
                    
                    # 1. Capture the raw output from AI (on Magenta BG)
                    raw_art = sq_result.crop((pad_x, pad_y, pad_x + target_w, pad_y + target_h))
                    
                    # 2. [Auto-Crop] Find the actual content bounding box (Dynamic Background Removal)
                    # Convert to RGB to check color distance
                    rgb_art = raw_art.convert("RGB")
                    np_art = np.array(rgb_art)
                    h_art, w_art, _ = np_art.shape
                    
                    # [Technique] Sample Background Color from Corners
                    # Instead of hardcoded Magenta, we sample the 4 corners to find the AI's actual background color.
                    # AI might shift the color slightly (e.g. 250, 10, 250).
                    corners = [
                        np_art[0, 0],
                        np_art[0, w_art-1],
                        np_art[h_art-1, 0],
                        np_art[h_art-1, w_art-1]
                    ]
                    bg_mean = np.mean(corners, axis=0) # [R, G, B]
                    print(f">>> [Auto-Crop] Detected Background Color: {bg_mean.astype(int)}")
                    
                    # Calculate distance from the detected background color
                    # Euclidean distance in RGB space
                    diff = np_art - bg_mean
                    dist = np.sqrt(np.sum(diff**2, axis=2))
                    
                    # Threshold: Allow variance. Pixels with dist > 70 are "Content".
                    # Increased threshold to be stricter about what is "Content".
                    mask = (dist > 70).astype(np.uint8) * 255
                    
                    # [Refinement] Remove Purple Halo
                    # Erode the mask slightly to cut off the semi-transparent purple edges.
                    kernel = np.ones((3,3), np.uint8)
                    mask = cv2.erode(mask, kernel, iterations=1)
                    
                    # Optional: Soften the edge slightly after erosion
                    # mask = cv2.GaussianBlur(mask, (3, 3), 0)
                    
                    mask_pil = Image.fromarray(mask, mode="L")
                    
                    bbox = mask_pil.getbbox()
                    
                    if bbox:
                        print(f">>> [Auto-Resize] Content found at {bbox}. Cropping and Resizing...")
                        cropped_art = raw_art.crop(bbox)
                        
                        # Apply mask to cropped art to make background transparent
                        cropped_mask = mask_pil.crop(bbox)
                        cropped_art.putalpha(cropped_mask)
                        
                        # 3. [Smart Resize] Fit to ACTUAL PANEL Size (target_main geometry)
                        # We use the raw panel dimensions, not the bleed 'mw, mh'
                        panel_w = int(target_main['w'])
                        panel_h = int(target_main['h'])
                        
                        # Apply 95% fill factor (Small margin inside dotted lines)
                        target_w = int(panel_w * 0.95)
                        target_h = int(panel_h * 0.95)
                        
                        # Current dimensions
                        cw, ch = cropped_art.size
                        
                        # Calculate scale to fit within target
                        scale = min(target_w / cw, target_h / ch)
                        new_w = int(cw * scale)
                        new_h = int(ch * scale)
                        
                        resized_art = cropped_art.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        
                        # 4. [Center Alignment] Calculate position to paste
                        # Use target_main['x'], target_main['y'] as layout origin
                        origin_x = int(target_main['x'])
                        origin_y = int(target_main['y'])
                        
                        center_x = origin_x + (panel_w - new_w) // 2
                        center_y = origin_y + (panel_h - new_h) // 2
                        
                        # Paste using alpha channel
                        print(f">>> [4단계] 메인 패널: 디자인 요소 자동 맞춤 합성 (Size: {new_w}x{new_h})")
                        final_canvas.paste(resized_art, (center_x, center_y), resized_art)
                    else:
                        print(">>> [Warning] No content found (Background removal failed). Pasting original.")
                        # Fallback: Just paste raw
                        final_canvas.paste(raw_art, (int(target_main['x']), int(target_main['y'])))

    except Exception as e:
        print(f">>> [Main Panel Error] {e}")

    # ==============================================================================
    # [3단계] 최종 합성 (Final Compositing)
    # - 생성된 이미지를 도면 외곽선에 맞춰 마스킹(Masking)하고
    # - 흐릿해진 도면 선을 선명하게 복원하여 합성합니다.
    # ==============================================================================
    print(">>> [5단계] 도면 마스킹 및 선명도 복원 (Crisp Dieline)...")
    
    # [Fix] 전개도 내부 패널만 정밀 마스킹 (Flood-fill 방식)
    # 구조 분석(Step 1)과 동일한 로직(INV Threshold + Dilate)을 사용하여 패널을 확실하게 분리합니다.
    gray_dieline = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 1. 선을 검출 (Black Lines -> White Lines)
    _, thresh = cv2.threshold(gray_dieline, 235, 255, cv2.THRESH_BINARY_INV)
    
    # 2. 선을 확장하여 끊어진 부분 연결 (Dilation)
    # Step 1보다는 얇게 확장하여 마스크 정밀도 확보 (Kernel 3x3 ~ 5x5)
    kernel = np.ones((5, 5), np.uint8)
    walled = cv2.dilate(thresh, kernel, iterations=1)
    
    # 3. 반전 (Lines=Black, Panels=White)하여 컴포넌트 추출
    num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(cv2.bitwise_not(walled))
    print(f">>> [Debug] Mask Generation (Robust): Found {num_labels} components")

    # (0,0) 좌표가 포함된 컴포넌트는 '배경'으로 간주하고 제외
    bg_label = labels_map[0, 0]
    
    mask = np.zeros(labels_map.shape, dtype=np.uint8)
    
    for i in range(1, num_labels):
        if i == bg_label: continue
        
        # 면적이 너무 작은 노이즈 제외
        if stats[i, cv2.CC_STAT_AREA] < (total_area * 0.0001):
            continue
            
        mask[labels_map == i] = 255
    
    # [Fix] Mask Dilation to remove white borders around lines
    # Dilate the mask to cover the gaps (lines) between panels.
    # This ensures the texture "bleeds" under the black dieline, removing white artifacts.
    dilation_kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, dilation_kernel, iterations=2)
    
    mask_pil = Image.fromarray(mask).convert("L")
    white_bg = Image.new("RGB", (w, h), (255, 255, 255))
    result_img = Image.composite(final_canvas.convert("RGB"), white_bg, mask_pil)
    
    # [Dieline Enhancement] 도면 선 복원 로직
    dieline_orig = Image.open(dieline_path).convert("L")
    
    # 1. Contrast: 선을 더 진하게
    enhancer = ImageEnhance.Contrast(dieline_orig)
    dieline_enhanced = enhancer.enhance(5.0) 
    
    # 2. Thresholding: 확실한 선만 남기고 나머지는 흰색으로 날림
    dieline_sharp = dieline_enhanced.point(lambda p: p if p < 200 else 255)
    
    # 3. Multiply Blend: 원본 이미지에 선을 '곱하기' 모드로 합성
    dieline_final = dieline_sharp.convert("RGB")
    final_output = ImageChops.multiply(result_img, dieline_final)

    output_file = "dieline_result.png"
    final_output.save(output_file)
    print(f"\n>>> [완료] 최종 결과물이 저장되었습니다: {output_file}")
    
    if platform.system() == "Darwin": subprocess.run(["open", output_file])
    elif platform.system() == "Windows": os.startfile(output_file)

if __name__ == "__main__":
    run_final_natural_pipeline("dieline_input.png", "concept_input.jpg")