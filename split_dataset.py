import os
import random
import albumentations as A
import cv2

# Number of augmentations per image
num_augmentations = 3  # Change this to increase/decrease augmentations

def augment_image(image):
    """ Apply augmentations to an image """
    transform = A.Compose([
        A.Resize(384, 384),  # Resize to 384x384
        A.HorizontalFlip(p=0.5),  # Flip half the images
        A.RandomBrightnessContrast(p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Rotate(limit=20, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),  # Adjust Hue & Saturation
    ])
    augmented = transform(image=image)
    return augmented["image"]  # Returns NumPy array

def split_dataset(dataset_dir, output_dir, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1, augment=True):
    """ Split dataset into train, valid, and test with optional augmentation """
    
    assert round(train_ratio + valid_ratio + test_ratio, 5) == 1.0, "Ratios must sum to 1"

    train_dir = os.path.join(output_dir, "train")
    valid_dir = os.path.join(output_dir, "valid")
    test_dir = os.path.join(output_dir, "test")

    for split in [train_dir, valid_dir, test_dir]:
        os.makedirs(split, exist_ok=True)

    # Iterate over model directories (since there are no brand directories)
    for model in os.listdir(dataset_dir):
        model_path = os.path.join(dataset_dir, model)
        if not os.path.isdir(model_path):
            continue  # Skip files, only process directories

        images = [f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))]
        random.shuffle(images)

        train_split = int(len(images) * train_ratio)
        valid_split = int(len(images) * (train_ratio + valid_ratio))

        train_images = images[:train_split]
        valid_images = images[train_split:valid_split]
        test_images = images[valid_split:]

        for split, split_images in zip([train_dir, valid_dir, test_dir], [train_images, valid_images, test_images]):
            split_model_dir = os.path.join(split, model)
            os.makedirs(split_model_dir, exist_ok=True)

            for img in split_images:
                src = os.path.join(model_path, img)
                dest = os.path.join(split_model_dir, img)

                image = cv2.imread(src)
                if image is None:
                    continue  # Skip corrupt images
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                cv2.imwrite(dest, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                if split == train_dir and augment:
                    for i in range(num_augmentations):  # Generate multiple augmentations per image
                        aug_img = augment_image(image)
                        aug_dest = os.path.join(split_model_dir, f"aug_{i}_{img}")  # Unique filename
                        cv2.imwrite(aug_dest, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))  # Save augmented image

    print(f"✅ Dataset successfully split and augmented! {num_augmentations} augmentations per training image.")

# Example Usage
dataset_root = "audi_dataset"  # Your dataset path
output_root = "dataset"  # Where the new dataset will be saved

split_dataset(dataset_root, output_root, augment=True)
