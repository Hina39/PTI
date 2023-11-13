import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
from configs import paths_config
from PIL import Image


def display_alongside_source_image(images):
    res = np.concatenate([np.array(image) for image in images], axis=1)
    return Image.fromarray(res)


def load_generators(model_id, image_name):
    with open(paths_config.stylegan2_ada_ffhq, "rb") as f:
        old_G = pickle.load(f)["G_ema"].cuda()

    with open(
        f"{paths_config.checkpoints_dir}/model_{model_id}_{image_name}.pt", "rb"
    ) as f_new:
        new_G = torch.load(f_new).cuda()

    return old_G, new_G


use_multi_id_training = False

model_id = "HEKOPCLKYMOE"
# image_name = "rekidai-index-97-abe"
# image_name = "IMG_4011"
# image_name = "10380049_598442736930128_1821329686642729572_o"
# image_name = "54277982_2068549176586136_5184358897338548224_n"
image_name = "47682676_1932313126876409_3822453097381232640_n"
generator_type = (
    paths_config.multi_id_model_type if use_multi_id_training else image_name
)
old_G, new_G = load_generators(model_id, generator_type)


import matplotlib.pyplot as plt


def plot_and_save_syn_images(syn_images, filenames):
    for img, filename in zip(syn_images, filenames):
        img = (
            (img.permute(0, 2, 3, 1).squeeze() * 127.5 + 128)
            .clamp(0, 255)
            .to(torch.uint8)
            .detach()
            .cpu()
            .numpy()
        )
        plt.imshow(img)
        plt.axis("off")
        plt.savefig(filename)
        plt.show()
        torch.cuda.empty_cache()


w_path_dir = f"{paths_config.embedding_base_dir}/{paths_config.input_data_id}"
embedding_dir = f"{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}"
w_pivot = torch.load(f"{embedding_dir}/0.pt")

old_image = old_G.synthesis(w_pivot, noise_mode="const", force_fp32=True)
new_image = new_G.synthesis(w_pivot, noise_mode="const", force_fp32=True)

print(
    "Upper image is the inversion before Pivotal Tuning and the lower image is the product of pivotal tuning"
)
plot_and_save_syn_images(
    [old_image, new_image],
    [f"./outputs/{image_name}_old.png", f"./outputs/{image_name}_new.png"],
)

# InterfaceGAN edits
from scripts.latent_editor_wrapper import LatentEditorWrapper

latent_editor = LatentEditorWrapper()
latents_after_edit = latent_editor.get_single_interface_gan_edits(w_pivot, [-4, 4])

for direction, factor_and_edit in latents_after_edit.items():
    print(f"Showing {direction} change")
    for latent in factor_and_edit.values():
        old_image = old_G.synthesis(latent, noise_mode="const", force_fp32=True)
        new_image = new_G.synthesis(latent, noise_mode="const", force_fp32=True)
        plot_and_save_syn_images(
            [old_image, new_image],
            [
                f"./outputs/{image_name}_old_edit.png",
                f"./outputs/{image_name}_new_edit.png",
            ],
        )


# old_Gとnew_Gは、それぞれ元のジェネレータと新しいジェネレータを表しています。
# これらは、通常、生成的敵対ネットワーク（GAN）の一部で、ランダムなノイズから新しい画像を生成する能力があります。

# このコードでは、old_Gは元のモデルからロードされ、new_Gは訓練後のモデルからロードされます。
# これらのジェネレータは、同じ入力（ここではw_pivot）を使用して画像を生成します。
# その結果、old_Gは訓練前の画像を生成し、new_Gは訓練後の画像を生成します。

# これらの画像は、plot_and_save_syn_images関数によって描画および保存されます。
# 具体的には、old_image.pngとnew_image.pngという名前のファイルに保存されます。
