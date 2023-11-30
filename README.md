# Classez des Images à l'aide d'Algorithmes de Deep Learning

# Technology recherchées et/ou utilisée

- Pytorch
- Torchvision

- Data Augmentation:

  - [AutoAugment](https://arxiv.org/pdf/1805.09501.pdf)
  - [Learning Data Augmentation](https://arxiv.org/pdf/1906.11172.pdf)
  - [Fast AutoAugment](https://arxiv.org/pdf/1905.00397.pdf)
  - [Randaugment](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.pdf)

- Architectures:

  - Single Shot:
    - YOLO
    - SSD
  - Two Stage:
    - Faster R-CNN
    - Mask R-CNN
  - Anchor Based
    - RetinaNet
    - SSD
  - Anchor Free
    - CenterNet
    - EfficientDet

- Pistes de recherche:
  - Feature extractions:
    - Scale Invariant Feature T
    - Harris Corner Detector
    - ORB : https://www.gwylab.com/download/ORB_2012.pdf
  - Mixup : https://arxiv.org/abs/1710.09412v2
  - Mosaic : https://iopscience.iop.org/article/10.1088/1742-6596/1684/1/012094/pdf
  - Mosaic guillotine split : http://pds25.egloos.com/pds/201504/21/98/RectangleBinPack.pdf

# Tâches

- [x] Preprocessing
  - [x] Equalization
  - [x] Whitening
- [x] Réaliser un réseau CNN manuellement
- [x] Réaliser un transfer learning d'un CNN
  - [x] Dernières couches
  - [x] Choisir des couches
- [x] Data augmentations
- [ ] Programme simple
