import gradio as gr
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
import csv
import json
import tempfile
import os
from datetime import datetime


# SAMモデルの初期化
def initialize_sam():
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


predictor = initialize_sam()


# グローバル状態の管理
class AnnotationState:
    def __init__(self):
        self.current_image = None
        self.current_masks = []
        self.current_scores = []
        self.annotations = []
        self.selected_mask_idx = None
        self.mode = "ai"  # 'ai' or 'manual'
        self.manual_points = []  # 手動モードでの頂点リスト

    def reset(self):
        self.__init__()


state = AnnotationState()


def mask_to_polygon(mask):
    """マスクから輪郭を抽出してpolygon形式に変換"""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return []

    # 最大の輪郭を選択
    largest_contour = max(contours, key=cv2.contourArea)

    # 点群を[(x, y), ...]形式に変換
    polygon = [(int(point[0][0]), int(point[0][1])) for point in largest_contour]

    return polygon


def polygon_to_mask(polygon, image_shape):
    """多角形からマスクを生成"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if len(polygon) < 3:
        return mask.astype(bool)

    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def get_mask_center(mask):
    """マスクの中心座標を計算"""
    y_coords, x_coords = np.where(mask)
    if len(x_coords) == 0:
        return 0, 0
    center_x = int(np.mean(x_coords))
    center_y = int(np.mean(y_coords))
    return center_x, center_y


def draw_polygon_preview(image, points):
    """描画中の多角形をプレビュー表示"""
    if image is None or len(points) == 0:
        return image

    img = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # 頂点を描画
    for point in points:
        x, y = point
        draw.ellipse(
            [x - 5, y - 5, x + 5, y + 5],
            fill=(255, 0, 0, 255),
            outline=(255, 255, 255, 255),
        )

    # 線を描画
    if len(points) > 1:
        draw.line(points, fill=(255, 255, 0, 200), width=2)

    # 最初の点と最後の点を結ぶ線（多角形の閉じる予定の線）
    if len(points) > 2:
        draw.line([points[-1], points[0]], fill=(255, 255, 0, 100), width=2)

    result = Image.alpha_composite(img, overlay)
    return result.convert("RGB")


def visualize_annotations(image, annotations, preview_points=None):
    """現在のアノテーション一覧を可視化"""
    if image is None:
        return None

    # 色のパレット
    colors = [
        (255, 0, 0, 128),  # 赤
        (0, 255, 0, 128),  # 緑
        (0, 0, 255, 128),  # 青
        (255, 255, 0, 128),  # 黄
        (255, 0, 255, 128),  # マゼンタ
        (0, 255, 255, 128),  # シアン
    ]

    result = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for i, ann in enumerate(annotations):
        color = colors[i % len(colors)]
        mask = ann["mask"]
        label = ann["label"]
        center = ann["center"]

        # マスクを描画
        mask_overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
        mask_overlay[mask] = color
        mask_img = Image.fromarray(mask_overlay, mode="RGBA")
        overlay = Image.alpha_composite(overlay, mask_img)

        # ラベルテキストを描画
        draw = ImageDraw.Draw(overlay)
        font = ImageFont.load_default(24)
        draw.text(
            center,
            f"{i + 1}: {label}",
            fill=(255, 255, 255, 255),
            font=font,
        )

    result = Image.alpha_composite(result, overlay)

    # プレビュー中の多角形を描画
    if preview_points and len(preview_points) > 0:
        draw = ImageDraw.Draw(result)
        for point in preview_points:
            x, y = point
            draw.ellipse(
                [x - 5, y - 5, x + 5, y + 5],
                fill=(255, 0, 0, 255),
                outline=(255, 255, 255, 255),
            )

        if len(preview_points) > 1:
            draw.line(preview_points, fill=(255, 255, 0, 255), width=2)

        if len(preview_points) > 2:
            draw.line(
                [preview_points[-1], preview_points[0]],
                fill=(255, 255, 0, 128),
                width=2,
            )

    return result.convert("RGB")


def upload_image(image):
    """画像アップロード時の処理"""
    state.reset()
    if image is not None:
        state.current_image = np.array(image)
        predictor.set_image(state.current_image)
    return (
        image,
        None,
        "画像をクリックしてセグメント化したい領域を選択してください",
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def change_mode(mode):
    """モード変更"""
    state.mode = mode
    state.manual_points = []
    state.selected_mask_idx = None

    if mode == "ai":
        return (
            "AIモード: 画像上をクリックしてセグメンテーションを実行します",
            gr.update(visible=False),
            gr.update(visible=False),
            visualize_annotations(state.current_image, state.annotations),
        )
    else:
        return (
            "手動モード: 画像上をクリックして多角形の頂点を設定します",
            gr.update(visible=True),
            gr.update(visible=True),
            visualize_annotations(state.current_image, state.annotations),
        )


def on_image_click(image, evt: gr.SelectData):
    """画像クリック時の処理（モードによって分岐）"""
    if state.current_image is None:
        return (
            None,
            "先に画像をアップロードしてください",
            gr.update(visible=False),
            None,
        )

    x, y = evt.index[0], evt.index[1]

    if state.mode == "ai":
        # AIモード: セグメンテーション実行
        input_point = np.array([[x, y]])
        input_label = np.array([1])

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        state.current_masks = masks
        state.current_scores = scores

        # 結果の可視化
        results = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            colored_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)
            colored_mask[mask] = [255, 0, 0, 128]

            result_image = state.current_image.copy()
            mask_img = Image.fromarray(colored_mask, mode="RGBA")
            base_img = Image.fromarray(result_image).convert("RGBA")
            combined = Image.alpha_composite(base_img, mask_img)

            results.append(combined.convert("RGB"))

        return (
            results,
            "セグメンテーション結果から最適なものを選択してください",
            gr.update(visible=True),
            None,
        )

    else:
        # 手動モード: 頂点を追加
        state.manual_points.append((x, y))
        preview_image = visualize_annotations(
            state.current_image, state.annotations, state.manual_points
        )

        return (
            None,
            f"頂点 {len(state.manual_points)} を追加しました。続けてクリックするか、「多角形を完成」ボタンを押してください",
            gr.update(visible=False),
            preview_image,
        )


def complete_polygon():
    """手動モードで多角形を完成させる"""
    if len(state.manual_points) < 3:
        return None, "最低3つの頂点が必要です", None

    # 多角形からマスクを生成
    mask = polygon_to_mask(state.manual_points, state.current_image.shape)

    # マスクを可視化
    colored_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)
    colored_mask[mask] = [255, 0, 0, 128]

    result_image = state.current_image.copy()
    mask_img = Image.fromarray(colored_mask, mode="RGBA")
    base_img = Image.fromarray(result_image).convert("RGBA")
    combined = Image.alpha_composite(base_img, mask_img)

    # 状態に保存
    state.current_masks = [mask]
    state.selected_mask_idx = 0

    return (
        combined.convert("RGB"),
        "多角形が完成しました。ラベルを入力して追加してください",
        visualize_annotations(state.current_image, state.annotations),
    )


def cancel_polygon():
    """手動モードで描画中の多角形をキャンセル"""
    state.manual_points = []
    preview_image = visualize_annotations(state.current_image, state.annotations)
    return preview_image, "描画をキャンセルしました"


def select_mask(evt: gr.SelectData):
    """マスク候補を選択（AIモードのみ）"""
    state.selected_mask_idx = evt.index
    return f"マスク {evt.index + 1} を選択しました。ラベルを入力してください"


def add_annotation(label_text):
    """アノテーションを追加"""
    if state.selected_mask_idx is None:
        return (
            None,
            "先にマスクを選択するか、手動モードで多角形を完成させてください",
            gr.update(value=""),
            None,
        )

    if not label_text or label_text.strip() == "":
        return None, "ラベルを入力してください", gr.update(value=""), None

    # 選択されたマスクを取得
    mask = state.current_masks[state.selected_mask_idx]

    # マスク情報を計算
    center = get_mask_center(mask)

    if state.mode == "manual":
        # 手動モードでは既存の頂点をそのまま使用
        polygon = state.manual_points
    else:
        # AIモードではマスクから輪郭を抽出
        polygon = mask_to_polygon(mask)

    # アノテーションを追加
    annotation = {
        "mask": mask,
        "label": label_text.strip(),
        "center": center,
        "polygon": polygon,
    }
    state.annotations.append(annotation)

    # 状態をリセット
    state.selected_mask_idx = None
    state.manual_points = []

    # 可視化を更新
    annotated_image = visualize_annotations(state.current_image, state.annotations)

    mode_text = "AI" if state.mode == "ai" else "手動"

    return (
        annotated_image,
        f"ラベル '{label_text}' を追加しました（全{len(state.annotations)}件, {mode_text}モード）。次の領域を選択してください",
        gr.update(value=""),
        annotated_image,
    )


def export_annotations():
    """アノテーションをCSV形式でエクスポート"""
    if len(state.annotations) == 0:
        return None, "エクスポートするアノテーションがありません"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        suffix=f"_annotations_{timestamp}.csv",
        newline="",
        encoding="utf-8",
    )

    try:
        writer = csv.writer(temp_file)

        # ヘッダー
        writer.writerow(["no", "type", "center_x", "center_y", "polygon"])

        # データ行
        for i, ann in enumerate(state.annotations, 1):
            polygon_str = json.dumps(ann["polygon"])
            writer.writerow(
                [i, ann["label"], ann["center"][0], ann["center"][1], polygon_str]
            )

        temp_file.close()

        return (
            temp_file.name,
            f"{len(state.annotations)}件のアノテーションをエクスポートしました",
        )

    except Exception as e:
        if temp_file:
            temp_file.close()
        return None, f"エクスポート中にエラーが発生しました: {str(e)}"


def clear_all():
    """すべてクリア"""
    state.reset()
    return (
        None,
        None,
        "新しい画像をアップロードしてください",
        gr.update(visible=False),
        gr.update(value=""),
        None,
        gr.update(visible=False),
        gr.update(visible=False),
    )


# Gradio UI構築
with gr.Blocks(title="SAM Interactive Annotation Tool") as demo:
    gr.Markdown("# SAM Interactive Annotation Tool")
    # 折りたたみ可能な使い方セクション
    with gr.Accordion("📖 使い方ガイド", open=False):
        gr.Markdown("""
        ### AIモード
        1. 画像をアップロード
        2. セグメント化したい領域をクリック
        3. 表示された候補から最適なマスクを選択
        4. ラベルを入力して追加

        ### 手動モード
        1. 画像をアップロード
        2. モードを「手動」に切り替え
        3. 多角形の頂点をクリックで設定（3点以上）
        4. 「多角形を完成」ボタンをクリック
        5. ラベルを入力して追加

        ### 共通
        - 2-5を繰り返してすべての領域にラベル付け
        - 完了したらExportボタンでCSVをダウンロード
        """)

    # モード選択
    mode_radio = gr.Radio(
        choices=["ai", "manual"],
        value="ai",
        label="アノテーションモード",
        info="AI: SAMによる自動セグメンテーション / 手動: 多角形を手動で描画",
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="1. 画像をアップロード", type="pil")

            status_text = gr.Textbox(
                label="ステータス",
                value="画像をアップロードしてください",
                interactive=False,
            )

            # 手動モード用ボタン
            with gr.Row(visible=False) as manual_buttons:
                complete_polygon_btn = gr.Button("多角形を完成", variant="primary")
                cancel_polygon_btn = gr.Button("キャンセル", variant="secondary")

            with gr.Row():
                clear_btn = gr.Button("すべてクリア", variant="secondary")
                export_btn = gr.Button("Export CSV", variant="primary")

            csv_output = gr.File(label="エクスポートされたCSV")

        with gr.Column(scale=1):
            annotated_display = gr.Image(label="アノテーション済み画像", type="pil")

    mask_gallery = gr.Gallery(
        label="2. セグメンテーション結果（最適なものを選択）",
        columns=3,
        height="auto",
        visible=False,
    )

    manual_preview = gr.Image(label="多角形プレビュー", type="pil", visible=False)

    with gr.Row():
        label_input = gr.Textbox(
            label="3. ラベル名を入力", placeholder="例: person, car, tree"
        )
        add_label_btn = gr.Button("4. ラベルを追加", variant="primary")

    # イベント設定
    input_image.upload(
        fn=upload_image,
        inputs=[input_image],
        outputs=[
            input_image,
            annotated_display,
            status_text,
            mask_gallery,
            label_input,
            manual_buttons,
            manual_preview,
        ],
    )

    mode_radio.change(
        fn=change_mode,
        inputs=[mode_radio],
        outputs=[status_text, manual_buttons, manual_preview, annotated_display],
    )

    input_image.select(
        fn=on_image_click,
        inputs=[input_image],
        outputs=[mask_gallery, status_text, mask_gallery, annotated_display],
    )

    mask_gallery.select(fn=select_mask, outputs=[status_text])

    complete_polygon_btn.click(
        fn=complete_polygon, outputs=[manual_preview, status_text, annotated_display]
    )

    cancel_polygon_btn.click(
        fn=cancel_polygon, outputs=[annotated_display, status_text]
    )

    add_label_btn.click(
        fn=add_annotation,
        inputs=[label_input],
        outputs=[annotated_display, status_text, label_input, manual_preview],
    )

    export_btn.click(fn=export_annotations, outputs=[csv_output, status_text])

    clear_btn.click(
        fn=clear_all,
        outputs=[
            input_image,
            annotated_display,
            status_text,
            mask_gallery,
            label_input,
            csv_output,
            manual_buttons,
            manual_preview,
        ],
    )

if __name__ == "__main__":
    demo.launch()
