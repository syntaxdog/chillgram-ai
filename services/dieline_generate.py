import cv2
import json
import numpy as np
from pathlib import Path


class DielineAnalyzer:
    def __init__(self):
        # 파라미터 초기화 (필요시 튜닝 가능하도록 클래스 변수로 관리)
        self.ADAPT_BLOCK = 31
        self.ADAPT_C = 8
        self.MIN_DASH_AREA = 6
        self.MAX_DASH_AREA = 450
        self.MAX_DASH_W = 60
        self.MAX_DASH_H = 30
        self.MIN_DASH_W = 3
        self.MIN_DASH_H = 1
        self.CLUSTER_TOL_Y = 8
        self.CLUSTER_TOL_X = 8
        self.MIN_POINTS_ON_LINE = 12
        self.TOP_BODY_Y_RANGE = (0.15, 0.55)
        self.BOT_BODY_Y_RANGE = (0.45, 0.92)
        self.COVER_RATIO_MIN = 0.70

    def binarize(self, img_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        bw = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.ADAPT_BLOCK,
            self.ADAPT_C,
        )
        # ... (기존 침식/팽창 로직 동일) ...
        k_break_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        k_break_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        k_restore = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        bw_h = cv2.erode(bw, k_break_h, iterations=1)
        bw_h = cv2.dilate(bw_h, k_restore, iterations=1)
        bw_v = cv2.erode(bw, k_break_v, iterations=1)
        bw_v = cv2.dilate(bw_v, k_restore, iterations=1)
        bw_break = cv2.bitwise_or(bw_h, bw_v)
        k_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.morphologyEx(bw_break, cv2.MORPH_OPEN, k_clean, iterations=1)

    def cluster_1d(self, values, tol: int):
        if not values:
            return []
        values = sorted(values)
        groups = [[values[0]]]
        for v in values[1:]:
            if abs(v - groups[-1][-1]) <= tol:
                groups[-1].append(v)
            else:
                groups.append([v])
        return [(int(np.mean(g)), g) for g in groups]

    def pick_front_back(self, panels, bw, y_top, y_bottom):
        # ... (기존 로직 동일, self 참조 없음) ...
        h, w = bw.shape[:2]
        widths = [p["x2"] - p["x1"] for p in panels]
        med = np.median(widths) if widths else 0
        keep = list(range(len(panels)))
        if len(panels) >= 3 and med > 0:
            if widths[0] < 0.65 * med:
                keep = keep[1:]
            if widths[-1] < 0.65 * med:
                keep = keep[:-1]

        body_panels = [panels[i] for i in keep] if keep else panels
        if not body_panels:
            return None, None

        body_widths = [p["x2"] - p["x1"] for p in body_panels]
        order = np.argsort(body_widths)[::-1]
        cand = (
            [body_panels[order[0]], body_panels[order[1]]]
            if len(order) >= 2
            else [body_panels[order[0]]]
        )

        def flap_score(p):
            x1, x2 = int(p["x1"]), int(p["x2"])
            y1t, y2t = max(0, y_top - int(0.18 * h)), y_top
            y1b, y2b = y_bottom, min(h, y_bottom + int(0.20 * h))
            top_d = np.mean(bw[y1t:y2t, x1:x2] > 0) if y2t > y1t else 0.0
            bot_d = np.mean(bw[y1b:y2b, x1:x2] > 0) if y2b > y1b else 0.0
            return float(top_d + bot_d)

        if len(cand) == 2:
            s0, s1 = flap_score(cand[0]), flap_score(cand[1])
            front = cand[0] if s0 >= s1 else cand[1]
            back = cand[1] if front is cand[0] else cand[0]
            return front, back
        return cand[0], None

    def draw_front_back(self, img_bgr, front, back, y_top, y_bottom):
        out = img_bgr.copy()

        def draw_panel(p, color, label):
            if p:
                cv2.rectangle(
                    out,
                    (int(p["x1"]), int(y_top)),
                    (int(p["x2"]), int(y_bottom)),
                    color,
                    3,
                )
                cv2.putText(
                    out,
                    label,
                    (int(p["x1"]) + 10, int(y_top) + 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    2,
                )

        draw_panel(front, (0, 255, 0), "FRONT")
        draw_panel(back, (0, 0, 255), "BACK")
        return out

    # [핵심] 외부에서 호출할 진입점 함수
    def analyze(self, image_path: str, output_dir: Path):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError("이미지를 읽을 수 없습니다.")

        bw = self.binarize(img)
        h, w = bw.shape[:2]

        # 1. Connected Components
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            bw, connectivity=8
        )
        dash_points = []
        for i in range(1, nlabels):
            _, _, ww, hh, area = stats[i]
            if (
                self.MIN_DASH_AREA <= area <= self.MAX_DASH_AREA
                and self.MIN_DASH_W <= ww <= self.MAX_DASH_W
                and self.MIN_DASH_H <= hh <= self.MAX_DASH_H
            ):
                dash_points.append(centroids[i])

        # 2. Horizontal Lines
        ys = [int(cy) for _, cy in dash_points]
        h_lines = []
        for y_mean, _ in self.cluster_1d(ys, self.CLUSTER_TOL_Y):
            pts = [p for p in dash_points if abs(p[1] - y_mean) <= self.CLUSTER_TOL_Y]
            if len(pts) >= self.MIN_POINTS_ON_LINE:
                xs = [p[0] for p in pts]
                h_lines.append(
                    {"y": int(y_mean), "count": len(pts), "span": max(xs) - min(xs)}
                )
        h_lines.sort(key=lambda x: (x["span"], x["count"]), reverse=True)

        # 3. Pick Body Top/Bottom
        top_range = (
            int(self.TOP_BODY_Y_RANGE[0] * h),
            int(self.TOP_BODY_Y_RANGE[1] * h),
        )
        bot_range = (
            int(self.BOT_BODY_Y_RANGE[0] * h),
            int(self.BOT_BODY_Y_RANGE[1] * h),
        )

        tops = [l for l in h_lines if top_range[0] <= l["y"] <= top_range[1]]
        bots = [l for l in h_lines if bot_range[0] <= l["y"] <= bot_range[1]]

        if not tops or not bots:
            raise ValueError("상/하단 점선을 찾을 수 없습니다.")
        y_top, y_bottom = tops[0]["y"], bots[0]["y"]
        if y_top > y_bottom:
            y_top, y_bottom = y_bottom, y_top
        BODY_H = max(1, y_bottom - y_top)

        # 4. Vertical Lines & Panels
        xs = [int(cx) for cx, _ in dash_points]
        v_lines = []
        for x_mean, _ in self.cluster_1d(xs, self.CLUSTER_TOL_X):
            pts = [p for p in dash_points if abs(p[0] - x_mean) <= self.CLUSTER_TOL_X]
            if len(pts) >= self.MIN_POINTS_ON_LINE:
                ys_pts = [p[1] for p in pts]
                cover = (min(max(ys_pts), y_bottom) - max(min(ys_pts), y_top)) / BODY_H
                if cover >= self.COVER_RATIO_MIN:
                    v_lines.append(int(x_mean))
        v_lines.sort()

        if len(v_lines) < 2:
            raise ValueError("패널 구분선을 충분히 찾지 못했습니다.")

        panels = []
        for i in range(len(v_lines) - 1):
            panels.append(
                {
                    "panel_index": i,
                    "x1": v_lines[i],
                    "x2": v_lines[i + 1],
                    "y1": y_top,
                    "y2": y_bottom,
                }
            )

        # 5. Front/Back & Save Images
        front, back = self.pick_front_back(panels, bw, y_top, y_bottom)

        # 결과 이미지 저장 (output_dir 사용)
        vis_fb = self.draw_front_back(img, front, back, y_top, y_bottom)
        cv2.imwrite(str(output_dir / "dieline_fb.png"), vis_fb)

        # 6. Return Result
        return {
            "y_top": int(y_top),
            "y_bottom": int(y_bottom),
            "panels": panels,
            "front": (
                {"x1": int(front["x1"]), "x2": int(front["x2"])} if front else None
            ),
            "back": {"x1": int(back["x1"]), "x2": int(back["x2"])} if back else None,
        }
