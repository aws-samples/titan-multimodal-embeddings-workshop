from PIL import Image


def resize_image(input_path, max_size = 2048):
    # Open the image
    image = Image.open(input_path)

    # Get the original width and height
    original_width, original_height = image.size

    # Check if resizing is necessary
    if original_width <= max_size and original_height <= max_size:
        #print(original_width)
        #print("Image is already within the acceptable size.")
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