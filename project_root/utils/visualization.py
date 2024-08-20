import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
import streamlit as st
import os

import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def visualize_segmented_objects(image, metadata_file_path, master_input_id):
    output_dir = "project_root/data/output"

    with open(metadata_file_path, 'r') as file:
        metadata = json.load(file)

    # Find the correct master image entry based on master_input_id
    master_image = next((item for item in metadata["master_image"] if item["id"] == master_input_id), None)

    if master_image is None:
        raise ValueError(f"Master image with ID {master_input_id} not found in metadata.")

    # Create a plot with the original image
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Add annotations for each object in the specified master image
    for obj in master_image["objects"]:
        for obj_data in obj["objects"]:
            bbox = obj_data["bbox"]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(bbox[0], bbox[1] - 10, obj_data['label'], color='red', fontsize=12, weight='bold')

    # Construct the path where the segmented image will be saved
    segmented_image_dir = os.path.join(output_dir, master_input_id)

    # Ensure the directory exists
    os.makedirs(segmented_image_dir, exist_ok=True)

    # Save the output image with annotations
    segmented_image_path = os.path.join(segmented_image_dir, "segmented_image.png")
    plt.axis('off')
    plt.savefig(segmented_image_path, bbox_inches='tight')
    plt.close()  # Close the plot to free up memory
    
    return segmented_image_path


def display_segmented_object(image_master_id):
    images_per_row = 3
    standard_size = (200, 200)  # Define the standard size for all images

    segmented_object_dir = f"project_root/data/segmented_objects/{image_master_id}"

    # Get a list of image files in the directory
    image_files = [os.path.join(segmented_object_dir, file) for file in os.listdir(segmented_object_dir)]

    # Display the images in a grid
    for i in range(0, len(image_files), images_per_row):
        cols = st.columns(images_per_row)
        for j, col in enumerate(cols):
            if i + j < len(image_files):
                image = Image.open(image_files[i + j])
                image = image.resize(standard_size)  # Resize the image to the standard size
                col.image(image, use_column_width=True)