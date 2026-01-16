import gradio as gr
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import torch
from PIL import Image


# SAMモデルの初期化
def initialize_sam():
    # モデルのチェックポイントをダウンロードしておく必要があります
    # https://github.com/facebookresearch/segment-anything#model-checkpoints
    sam_checkpoint = "sam_vit_h_4b8939.pth"  # vit_h, vit_l, vit_b
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


predictor = initialize_sam()


def segment_image(image, evt: gr.SelectData):
    """
    画像上でクリックされた位置をプロンプトとしてセグメンテーションを実行
    """
    # 画像をnumpy配列に変換
    if isinstance(image, Image.Image):
        image = np.array(image)

    # クリック位置の取得
    x, y = evt.index[0], evt.index[1]
    input_point = np.array([[x, y]])
    input_label = np.array([1])  # 1 = foreground point

    # SAMで画像をエンコード
    predictor.set_image(image)

    # セグメンテーション実行
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,  # 複数のマスク候補を返す
    )

    # 結果の可視化
    results = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        # マスクをRGBA画像に変換
        colored_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)
        colored_mask[mask] = [255, 0, 0, 128]  # 赤色、半透明

        # 元画像とマスクを合成
        result_image = image.copy()
        mask_img = Image.fromarray(colored_mask, mode="RGBA")
        base_img = Image.fromarray(result_image).convert("RGBA")
        combined = Image.alpha_composite(base_img, mask_img)

        results.append((combined, f"Mask {i+1} (Score: {score:.3f})"))

    return results


# Gradio UI構築
with gr.Blocks() as demo:
    gr.Markdown("# SAM Interactive Segmentation")
    gr.Markdown("画像をアップロードして、セグメント化したい対象をクリックしてください")

    with gr.Row():
        input_image = gr.Image(label="入力画像", type="pil")

    with gr.Row():
        gallery = gr.Gallery(label="セグメンテーション結果", columns=3, height="auto")

    # クリックイベントの設定
    input_image.select(fn=segment_image, inputs=[input_image], outputs=[gallery])

if __name__ == "__main__":
    demo.launch()
