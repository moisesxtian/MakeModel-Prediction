## THIS SCRIPT FIXES DATASET FORMAT TO BRAND/Model/img.jpg instead of Brand/Model/Model/Year/etc../img.jpg

import os
import shutil

def move_images_to_brand(root_dir):
    for make in os.listdir(root_dir):  # Iterate over car brands (e.g., Lincoln)
        make_path = os.path.join(root_dir, make)
        
        if not os.path.isdir(make_path):
            continue  # Skip non-directory files

        for model in os.listdir(make_path):  # Iterate over car models (e.g., Lincoln-LS)
            model_path = os.path.join(make_path, model)

            if not os.path.isdir(model_path):
                continue  # Skip non-directory files

            # Move all images from model and its subfolders to the brand folder
            for subfolder in os.listdir(model_path):
                subfolder_path = os.path.join(model_path, subfolder)
                full_path = os.path.join(model_path, subfolder)

                if os.path.isdir(full_path):  # If it's a folder, move its images
                    for img in os.listdir(full_path):
                        img_path = os.path.join(full_path, img)
                        new_path = os.path.join(make_path, img)  # Move to brand folder
                        
                        if os.path.isfile(img_path):  # Ensure it's a file
                            shutil.move(img_path, new_path)

                    os.rmdir(full_path)  # Remove the empty subfolder after moving files
            
            # After processing subfolders, move any remaining images in the model folder
            for img in os.listdir(model_path):
                img_path = os.path.join(model_path, img)
                new_path = os.path.join(make_path, img)

                if os.path.isfile(img_path):
                    shutil.move(img_path, new_path)

            os.rmdir(model_path)  # Remove the empty model folder

    print("✅ All images moved to brand folders successfully!")

# Example Usage
dataset_root = "DATASET"  # Change this to your dataset path
move_images_to_brand(dataset_root)
