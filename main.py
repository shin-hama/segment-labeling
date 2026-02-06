import gradio as gr
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import torch
import cv2
import csv
import tempfile
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
        self.current_image: None | np.ndarray = None
        self.current_masks: np.ndarray = np.array([])
        self.annotations = []
        self.selected_mask_idx: None | int = None
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


def draw_polygon_preview(image: None | np.ndarray, points):
    """描画中の多角形をプレビュー表示"""
    if image is None or len(points) == 0:
        return image

    # 元の画像をコピー（NumPy配列として処理）
    img = image.copy()

    imageSize = image.shape[1], image.shape[0]
    lineWidth = max(1, min(imageSize) // 200)

    # 線を描画
    if len(points) > 1:
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(img, [pts], False, (0, 255, 255), lineWidth)

    # 最初の点と最後の点を結ぶ線（薄い色で）
    if len(points) > 2:
        cv2.line(
            img, tuple(points[-1]), tuple(points[0]), (0, 255, 255), lineWidth // 2
        )

    # 頂点を描画
    for point in points:
        x, y = int(point[0]), int(point[1])
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # 塗りつぶし
        cv2.circle(img, (x, y), 5, (255, 255, 255), lineWidth)  # 外枠

    return img


def visualize_annotations(image: None | np.ndarray, annotations):
    """現在のアノテーション一覧を可視化（高速版）"""
    if image is None:
        return None

    # 色のパレット (BGR形式)
    colors = [
        np.array([0, 0, 255]),  # 赤
        np.array([0, 255, 0]),  # 緑
        np.array([255, 0, 0]),  # 青
        np.array([0, 255, 255]),  # 黄
        np.array([255, 0, 255]),  # マゼンタ
        np.array([255, 255, 0]),  # シアン
    ]

    result = image.copy()
    overlay = np.zeros_like(image)

    # アノテーションを描画
    for i, ann in enumerate(annotations):
        color = colors[i % len(colors)]
        mask = ann["mask"]
        label = ann["label"]
        center = ann["center"]

        # マスクを描画（overlayに色を塗る）
        overlay[mask] = color

        # ラベルテキストを描画
        imageSize = image.shape[1], image.shape[0]
        fontSize = min(imageSize) / 1000  # OpenCVのフォントサイズ
        thickness = max(1, int(min(imageSize) / 500))

        cv2.putText(
            result,
            f"{i + 1}: {label}",
            tuple(map(int, center)),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontSize,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        # オーバーレイを半透明で合成
        alpha = 0.5
        result[mask] = result[mask] * (1 - alpha) + color * alpha

    return result


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
            visualize_annotations(state.current_image, state.annotations),
        )
    else:
        return (
            "手動モード: 画像上をクリックして多角形の頂点を設定します",
            gr.update(visible=True),
            visualize_annotations(state.current_image, state.annotations),
        )


def on_image_click(image: None | np.ndarray, evt: gr.SelectData):
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
        state.selected_mask_idx = None

        # 結果の可視化
        results = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            result_image = state.current_image.copy().astype(np.float32)

            # マスク領域を赤色で半透明に塗る
            color = np.array([0, 0, 255], dtype=np.float32)  # BGR: 赤
            alpha = 0.5
            result_image[mask] = result_image[mask] * (1 - alpha) + color * alpha

            result_image = result_image.astype(np.uint8)
            results.append(result_image)

        return (
            results,
            "セグメンテーション結果から最適なものを選択してください",
            gr.update(visible=True),
            state.current_image,
        )

    else:
        # 手動モード: 頂点を追加
        state.manual_points.append((x, y))
        if len(state.manual_points) >= 3:
            # 多角形からマスクを生成
            mask = polygon_to_mask(state.manual_points, state.current_image.shape)

            state.current_masks = np.array([mask])
            state.selected_mask_idx = 0

        # オリジナル画像上にも編集中のポリゴンを表示
        original_with_polygon = draw_polygon_preview(
            state.current_image, state.manual_points
        )

        return (
            None,
            f"頂点 {len(state.manual_points)} を追加しました。続けてクリックするか、ラベル名を入力して「多角形を完成」ボタンを押してください。",
            gr.update(visible=False),
            original_with_polygon,
        )


def cancel_polygon():
    """手動モードで描画中の多角形をキャンセル"""
    state.manual_points = []
    preview_image = visualize_annotations(state.current_image, state.annotations)
    return (
        preview_image,
        "描画をキャンセルしました",
        state.current_image,
    )


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
            state.current_image,
            gr.update(visible=True),
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
        state.current_image,
        gr.update(value=None, visible=False),
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
        writer.writerow(
            [
                "no",
                "type",
                "center_x",
                "center_y",
                "polygon",
                "flip",
                "rotation",
                "fraction",
                "confidfence",
                "flip_confidence",
                "rotation_confidence",
            ]
        )

        # データ行
        for i, ann in enumerate(state.annotations, 1):
            polygon_str = str([tuple(inner) for inner in ann["polygon"]])
            writer.writerow(
                [
                    i,
                    ann["label"],
                    ann["center"][0],
                    ann["center"][1],
                    polygon_str,
                    1,
                    0,
                    0.9,
                    -1,
                    -1,
                    -1,
                ]
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
        4. ラベルを入力して追加

        ### 共通
        - 2-4を繰り返してすべての領域にラベル付け
        - 完了したらExportボタンでCSVをダウンロード
        """)

    status_text = gr.Textbox(
        label="ステータス",
        value="画像をアップロードしてください",
        interactive=False,
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="1. 画像をアップロード", type="pil")

            # モード選択
            mode_radio = gr.Radio(
                choices=["ai", "manual"],
                value="ai",
                label="アノテーションモード",
                info="AI: SAMによる自動セグメンテーション / 手動: 多角形を手動で描画",
            )

            # 手動モード用ボタン
            with gr.Row(visible=False) as manual_buttons:
                complete_polygon_btn = gr.Button("多角形を完成", variant="primary")
                cancel_polygon_btn = gr.Button("キャンセル", variant="secondary")

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

    with gr.Row():
        clear_btn = gr.Button("すべてクリア", variant="secondary")
        export_btn = gr.Button("Export CSV", variant="primary")

    csv_output = gr.File(label="エクスポートされたCSV")

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
        ],
    )

    mode_radio.change(
        fn=change_mode,
        inputs=[mode_radio],
        outputs=[status_text, manual_buttons, annotated_display],
    )

    input_image.select(
        fn=on_image_click,
        inputs=[input_image],
        outputs=[
            mask_gallery,
            status_text,
            mask_gallery,
            input_image,
        ],
    )

    mask_gallery.select(fn=select_mask, outputs=[status_text])

    cancel_polygon_btn.click(
        fn=cancel_polygon, outputs=[annotated_display, status_text, input_image]
    )

    add_label_btn.click(
        fn=add_annotation,
        inputs=[label_input],
        outputs=[
            annotated_display,
            status_text,
            label_input,
            input_image,
            mask_gallery,
        ],
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
        ],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
