import albumentations as albu
import cv2


def get_transforms(opt, max_pixel_value=255):
    if opt.imagenet_norm:
        imagenet_mean = 0.485, 0.456, 0.406
        imagenet_std = 0.229, 0.224, 0.225
        grayscale_coefs = 0.2989, 0.587, 0.114

        grayscale_mean = sum(m * c for m, c in zip(imagenet_mean, grayscale_coefs))
        grayscale_std = sum(m * c for m, c in zip(imagenet_std, grayscale_coefs))

        # imagenet norm stats converted to grayscale
        mean = [grayscale_mean]
        std = [grayscale_std]
    else:
        mean = [0.5]
        std = [0.5]

    train_ops = []
    if 'hflip' in opt.augs:
        train_ops.append(albu.HorizontalFlip(p=0.5))
    if 'vflip' in opt.augs:
        train_ops.append(albu.VerticalFlip(p=0.1))
    if 'shift_scale_rotate' in opt.augs:
        train_ops.append(albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, shift_limit=0.07, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0))

    data_transforms = {
        'train': albu.Compose(
            [
                *train_ops,
                albu.Resize(*opt.img_size, cv2.INTER_LINEAR),
                albu.Normalize(mean, std, max_pixel_value=max_pixel_value),
            ]
        ),
        'val': albu.Compose(
            [
                albu.Resize(*opt.img_size, cv2.INTER_LINEAR),
                albu.Normalize(mean, std, max_pixel_value=max_pixel_value),
            ]
        ),
    }
    return data_transforms
