import os
import json
import uuid


def read_yaml(file_path):
    """
    Reads a YAML file and returns its contents as a dictionary.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: The contents of the YAML file as a dictionary.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    return data


def write_yaml(file_path,data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def generate_unique_id():
    """
    Generates a unique ID for each image.

    Returns:
        str: A unique ID for the image.
    """
    return str(uuid.uuid4())


def save_input_image(image, image_id):
    input_image_dir = 'project_root/data/input_images'
    
    # Ensure the directory exists
    if not os.path.exists(input_image_dir):
        os.makedirs(input_image_dir)
        print(f"Created directory: {input_image_dir}")
    
    image_path = os.path.join(input_image_dir, f"{image_id}.png")
    
    print(f"Saving image to: {image_path}")
    image.save(image_path)

    if os.path.exists(image_path):
        print("Image saved successfully!")
    else:
        print("Failed to save image.")
    
    return image_path
