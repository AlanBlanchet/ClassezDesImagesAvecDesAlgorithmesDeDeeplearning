import albumentations as A
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader


class StanfordDogsAugmentations:
    def __init__(self):
        p = 0.06

        self.transforms = A.Compose(
            [
                A.Transpose(),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90(),
                A.OneOf(
                    [
                        A.AdvancedBlur(p=p, blur_limit=(1, 3)),
                        A.RandomFog(p=p, fog_coef_upper=0.7),
                        A.RandomRain(
                            p=p,
                            blur_value=4,
                            drop_length=10,
                            slant_lower=-1,
                            slant_upper=5,
                        ),
                        A.RandomSnow(p=p, brightness_coeff=1.5, snow_point_upper=0.2),
                        A.ZoomBlur(p=p, max_factor=1.2),
                    ],
                    p=0.4,
                ),
                # A.RandomScale(),
                A.RandomBrightnessContrast(p=p),
                # A.RandomCropFromBorders(p=p),
                A.RandomGamma(p=p),
                # A.OpticalDistortion(),
                A.ChannelDropout(p=p),
                A.ChannelShuffle(p=p),
                A.CLAHE(p=p),
                A.Sharpen(p=p),
                A.RingingOvershoot(p=p),
                A.Spatter(p=p),
            ],
            bbox_params=A.BboxParams(
                format="albumentations", label_fields=["class_labels"]
            ),
        )

    def __call__(self, image, bbox=None, labels=None) -> tuple:
        return self.transforms(
            image=image.numpy(),
            bboxes=bbox.numpy(),
            class_labels=labels,
        )


class BaseAugmentations:
    def __init__(self, config: dict, torch_dataset, train_idx):
        self.torch_dataset = torch_dataset
        self.train_idx = train_idx
        self.img_size = min(config["img_size"], 32)

        self.zca_matrix = None
        if config.get("whitening", False):
            self.zca_matrix = self._zca()

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def _zca(self):
        images = []
        for idx in self.train_idx:
            image = self.torch_dataset[idx]["image"].numpy() / 255
            images.append(A.resize(image, self.img_size, self.img_size).tolist())
        images = torch.tensor(images, device="cuda")
        print(f"{images.shape=}")

        imgs_f = images.flatten(start_dim=1)
        imgs_mean = imgs_f.mean(axis=0)
        print("Mean/Cov...")
        imgs_f -= imgs_mean
        imgs_cov = imgs_f.T.cov()

        print("SVD...")
        U, S, _ = torch.svd(imgs_cov)
        e = 0.1

        zca_mat = (U @ torch.diag(1.0 / torch.sqrt(S + e))) @ U.T
        print("ZCA shape = ", zca_mat.shape)
        return zca_mat

    def zca(self, image):
        image = image.to("cuda")
        shape = image.shape
        # print(f"{image.shape=}")
        image = image.flatten()
        # print(f"{image.shape=}")
        # print(f"{self.zca_matrix.shape=}")
        imgs_zca = (self.zca_matrix @ image.unsqueeze(dim=1)).T
        # print(f"{imgs_zca.shape=}")
        imgs_zca = (imgs_zca - imgs_zca.min()) / (imgs_zca.max() - imgs_zca.min())
        return imgs_zca.squeeze(dim=0).view(shape)

    def __call__(self, image):
        # print("=" * 20)
        if self.zca_matrix is not None:
            return self.zca(image)
        return self.normalize(image)
