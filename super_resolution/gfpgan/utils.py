import cv2
import os
import torch
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize

from .archs.gfpgan_bilinear_arch import GFPGANBilinear
from .archs.gfpganv1_arch import GFPGANv1
from .archs.gfpganv1_clean_arch import GFPGANv1Clean

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GFPGANer:
    """Helper for restoration with GFPGAN.

    It will detect and crop faces, and then resize the faces to 512x512.
    GFPGAN is used to restored the resized faces.
    The background is upsampled with the bg_upsampler.
    Finally, the faces will be pasted back to the upsample background image.

    Args:
        model_path (str): The path to the GFPGAN model. It can be urls (will first download it automatically).
        upscale (float): The upscale of the final output. Default: 2.
        arch (str): The GFPGAN architecture. Option: clean | original. Default: clean.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        bg_upsampler (nn.Module): The upsampler for the background. Default: None.
    """

    def __init__(
        self,
        model_path,
        upscale=2,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
        device=None,
    ):
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler

        # initialize model
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )
        # initialize the GFP-GAN
        if arch == "clean":
            self.gfpgan = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True,
            )
        elif arch == "bilinear":
            self.gfpgan = GFPGANBilinear(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True,
            )
        elif arch == "original":
            self.gfpgan = GFPGANv1(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=True,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True,
            )
        # initialize face helper
        # 使用到了facexlib包中的face restoration helper
        self.face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=self.device,
        )

        # 增加了一个model路径是网址时的处理，然而需要的model已经下载到本地，并没有用到
        if model_path.startswith("https://"):
            model_path = load_file_from_url(
                url=model_path,
                model_dir=os.path.join(ROOT_DIR, "gfpgan/weights"),
                progress=True,
                file_name=None,
            )
        loadnet = torch.load(model_path)  # 加载模型
        # 读取model并继续初始化
        if "params_ema" in loadnet:
            keyname = "params_ema"
        else:
            keyname = "params"
        self.gfpgan.load_state_dict(loadnet[keyname], strict=True)
        self.gfpgan.eval()
        self.gfpgan = self.gfpgan.to(self.device)

    @torch.no_grad()
    def enhance(self, img, has_aligned=False, only_center_face=False, paste_back=True):
        self.face_helper.clean_all()  # self.face_helper.clean_all()

        #  进行仿射变换，相似变化（截取）
        if has_aligned:  # the inputs are already aligned
            img = cv2.resize(img, (512, 512))
            self.face_helper.cropped_faces = [img]
        else:
            self.face_helper.read_image(img)
            # get face landmarks for each face
            self.face_helper.get_face_landmarks_5(
                only_center_face=only_center_face, eye_dist_threshold=5
            )
            # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
            # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
            # align and warp each face
            self.face_helper.align_warp_face()

        # face restoration
        for cropped_face in self.face_helper.cropped_faces:
            # prepare data 修复前再做一下处理：归一化等
            cropped_face_t = img2tensor(
                cropped_face / 255.0, bgr2rgb=True, float32=True
            )
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)
            # unsqueeze(0)，在第0个维度之前增加一个维度

            # 调用gfpgan对图像进行修复，失败的话输出错误提醒，并返回未修复的照片
            try:
                output = self.gfpgan(cropped_face_t, return_rgb=False)[0]
                # convert to image
                # 经过修复后图片矩阵的数字一定是float的。但是需要保存.png，所以图片的数字类型必须变到[0,255]的整数,
                restored_face = tensor2img(
                    output.squeeze(0), rgb2bgr=True, min_max=(-1, 1)
                )
            except RuntimeError as error:
                print(f"\tFailed inference for GFPGAN: {error}.")
                restored_face = cropped_face

            # 调用astype('uint8')会将小数部分直接截去，0-255之间的整数保留
            restored_face = restored_face.astype("uint8")
            self.face_helper.add_restored_face(restored_face)

        # 如果满足not has_aligned=True：修复前没有图像对齐
        # 和paste_back=True：设置了需要将修复好的人脸贴回原图；做以下操作：
        if not has_aligned and paste_back:
            # upsample the background
            if self.bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                # 使用RealESRGAN对输入的 {完整图片} 进行修复，提高分辨率
                bg_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0]
            else:
                bg_img = None

            self.face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = self.face_helper.paste_faces_to_input_image(
                upsample_img=bg_img
            )
            # 返回修复过程中得到的图像包括：修复前的人脸，修复后的人脸以及完整的修复图片
            return (
                self.face_helper.cropped_faces,
                self.face_helper.restored_faces,
                restored_img,
            )
        else:
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, None
