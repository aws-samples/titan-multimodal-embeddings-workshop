from PIL import Image
import os
import requests
import zipfile
import shutil

def resize_image(input_path, max_size = 2048):
    # Open the image
    image = Image.open(input_path)

    # Get the original width and height
    original_width, original_height = image.size

    # Check if resizing is necessary
    if original_width <= max_size and original_height <= max_size:
        print(original_width)
        print("Image is already within the acceptable size.")
        return

    # Calculate the aspect ratio
    aspect_ratio = original_width / original_height

    # Calculate the new dimensions while maintaining the aspect ratio
    if original_width > original_height:
        new_width = max_size
        new_height = int(max_size / aspect_ratio)
    else:
        new_width = int(max_size * aspect_ratio)
        new_height = max_size

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Save the resized image
    resized_image.save(input_path)

def process_zip(zip_url):
    # Download the zip file
    response = requests.get(zip_url)
    zip_filename = os.path.basename(zip_url)
    with open(zip_filename, "wb") as file:
        file.write(response.content)

    # Unzip the downloaded file
    with zipfile.ZipFile(zip_filename, "r") as zip_ref:
        extracted_files = zip_ref.namelist()
        zip_ref.extractall()

    # Create the preview-sdk folder
    os.makedirs("preview-sdk", exist_ok=True)

    # Copy the required files to the preview-sdk folder
    shutil.copy("boto3-1.29.4-py3-none-any.whl", "preview-sdk")
    shutil.copy("botocore-1.32.4-py3-none-any.whl", "preview-sdk")

    print("Files copied successfully to the preview-sdk folder.")

    # Delete the downloaded zip file
    os.remove(zip_filename)

    # Delete the extracted files and directories
    for item in extracted_files:
        extracted_path = os.path.join(os.getcwd(), item)
        if os.path.isfile(extracted_path):
            os.remove(extracted_path)
        elif os.path.isdir(extracted_path):
            shutil.rmtree(extracted_path)

    print("Cleanup completed.")