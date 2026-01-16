import csv
import json
import tempfile
from datetime import datetime
from typing import Optional

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw
from segment_anything import SamPredictor, sam_model_registry


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
        self.current_image: Optional[np.ndarray] = None
        self.current_masks: Optional[np.ndarray] = None
        self.current_scores: Optional[np.ndarray] = None
        self.annotations = []  # {mask, label, center, polygon}
        self.selected_mask_idx = None

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


def get_mask_center(mask):
    """マスクの中心座標を計算"""
    y_coords, x_coords = np.where(mask)
    if len(x_coords) == 0:
        return 0, 0
    center_x = int(np.mean(x_coords))
    center_y = int(np.mean(y_coords))
    return center_x, center_y


def visualize_annotations(image, annotations):
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
        draw.text(center, f"{i + 1}: {label}", fill=(255, 255, 255, 255))

    result = Image.alpha_composite(result, overlay)
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
    )


def segment_on_click(image, evt: gr.SelectData):
    """画像クリック時のセグメンテーション実行"""
    if state.current_image is None:
        return None, "先に画像をアップロードしてください", gr.update(visible=False)

    # クリック位置の取得
    x, y = evt.index[0], evt.index[1]
    input_point = np.array([[x, y]])
    input_label = np.array([1])

    # セグメンテーション実行
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # 状態に保存
    state.current_masks = masks
    state.current_scores = scores

    # 結果の可視化
    results = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        # マスクを可視化
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
    )


def select_mask(evt: gr.SelectData):
    """マスク候補を選択"""
    state.selected_mask_idx = evt.index
    return f"マスク {evt.index + 1} を選択しました。ラベルを入力してください"


def add_annotation(label_text):
    """アノテーションを追加"""
    if state.selected_mask_idx is None:
        return None, "先にマスクを選択してください", gr.update(value="")

    if state.current_masks is None:
        return None, "先にセグメンテーションを実行してください", gr.update(value="")

    if not label_text or label_text.strip() == "":
        return None, "ラベルを入力してください", gr.update(value="")

    # 選択されたマスクを取得
    mask = state.current_masks[state.selected_mask_idx]

    # マスク情報を計算
    center = get_mask_center(mask)
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

    # 可視化を更新
    annotated_image = visualize_annotations(state.current_image, state.annotations)

    return (
        annotated_image,
        f"ラベル '{label_text}' を追加しました（全{len(state.annotations)}件）。次の領域をクリックするか、完了したらExportしてください",
        gr.update(value=""),
    )


def export_annotations():
    """アノテーションをCSV形式でエクスポート"""
    if len(state.annotations) == 0:
        return None, "エクスポートするアノテーションがありません"

    # 一時ファイルを作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
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
    )


# Gradio UI構築
with gr.Blocks(title="SAM Interactive Annotation Tool") as demo:
    gr.Markdown("# SAM Interactive Annotation Tool")
    gr.Markdown(
        """
    ## 使い方
    1. 画像をアップロード
    2. セグメント化したい領域をクリック
    3. 表示された候補から最適なマスクを選択
    4. ラベルを入力して追加
    5. 2-4を繰り返し
    6. 完了したら Export CSV ボタンで CSV をダウンロード
    """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="1. 画像をアップロード", type="pil")

            status_text = gr.Textbox(
                label="ステータス",
                value="画像をアップロードしてください",
                interactive=False,
            )

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
        ],
    )

    input_image.select(
        fn=segment_on_click,
        inputs=[input_image],
        outputs=[mask_gallery, status_text, mask_gallery],
    )

    mask_gallery.select(fn=select_mask, outputs=[status_text])

    add_label_btn.click(
        fn=add_annotation,
        inputs=[label_input],
        outputs=[annotated_display, status_text, label_input],
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
        ],
    )

if __name__ == "__main__":
    demo.launch()
