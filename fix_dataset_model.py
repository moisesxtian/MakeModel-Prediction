import os
import shutil

def flatten_model_folders(root_dir):
    for make in os.listdir(root_dir):  # Iterate over brands (e.g., Honda)
        make_path = os.path.join(root_dir, make)

        if not os.path.isdir(make_path):
            continue  # Skip non-directory files

        for model in os.listdir(make_path):  # Iterate over models (e.g., Civic, Accord)
            model_path = os.path.join(make_path, model)

            if not os.path.isdir(model_path):
                continue  # Skip non-directory files

            # Move all images from subfolders into the model folder
            for subfolder in os.listdir(model_path):
                subfolder_path = os.path.join(model_path, subfolder)

                if os.path.isdir(subfolder_path):  # Ensure it's a folder
                    for img in os.listdir(subfolder_path):  # Iterate over images
                        img_path = os.path.join(subfolder_path, img)
                        new_path = os.path.join(model_path, img)  # Move to model folder

                        if os.path.isfile(img_path):  # Ensure it's a file
                            shutil.move(img_path, new_path)

                    os.rmdir(subfolder_path)  # Remove empty subfolder after moving files

    print("✅ Dataset structure updated successfully!")

# Example Usage
dataset_root = "DATASET"  # Change this to your dataset path
flatten_model_folders(dataset_root)
