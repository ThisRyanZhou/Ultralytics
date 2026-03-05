from PIL import Image
import os

root = "letters_dataset"   # main dataset folder

for folder in os.listdir(root):
    folder_path = os.path.join(root, folder)

    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):

                path = os.path.join(folder_path, filename)

                img = Image.open(path)

                img = img.convert("L")      # convert to grayscale
                img = img.resize((64, 64))  # resize image

                img.save(path)

print("All images converted to grayscale and resized to 64x64.")
