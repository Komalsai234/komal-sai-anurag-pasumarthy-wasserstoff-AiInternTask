import os
import numpy as np
from PIL import Image
import cv2
from project_root.utils.preprocessing import read_yaml, write_yaml, generate_unique_id

class IdentificationModel:
    def __init__(self):
        self.segmented_object_dir = 'project_root/data/segmented_objects'
        self.metadata_dir = 'metadata/'

        if not os.path.exists(self.segmented_object_dir):
            os.makedirs(self.segmented_object_dir)
            print(f"Created directory: {self.segmented_object_dir}")

        if not os.path.exists(self.metadata_dir):
            os.makedirs(self.metadata_dir)
            print(f"Created directory: {self.metadata_dir}")

    def identify_objects(self, results, label_ids_list):
        objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            objects.append({
                "label": label_ids_list[label.item()],
                "confidence": round(score.item(), 3),
                "bbox": box
            })
        return objects
    
    def segmented_object_metadata(self, objects, image, master_image_id):
        image_np = np.array(image)  # Convert to NumPy array

        os.makedirs(os.path.join(self.segmented_object_dir,master_image_id),exist_ok=True)

        objects_metadata = []

        for detection in objects:
            label = detection["label"]
            confidence = detection["confidence"]
            bbox = detection["bbox"]

            x1, y1, x2, y2 = map(int, bbox)

            # Adjust coordinates to be within image bounds
            x1 = max(0, min(x1, image_np.shape[1] - 1))
            y1 = max(0, min(y1, image_np.shape[0] - 1))
            x2 = max(0, min(x2, image_np.shape[1] - 1))
            y2 = max(0, min(y2, image_np.shape[0] - 1))

            extracted_object = image_np[y1:y2, x1:x2]  # Extract object using NumPy slicing

            object_id = generate_unique_id()
            output_image_path = os.path.join(self.segmented_object_dir,master_image_id, f"{object_id}.png")

            cv2.imwrite(output_image_path, extracted_object)

            metadata_entry = {
                "object_id": object_id,
                "label": label,
                "confidence": confidence,
                "bbox": bbox
            }

            objects_metadata.append(metadata_entry)

        metadata = {
            'id': master_image_id,
            'objects': objects_metadata
        }

        return metadata
    
    def save_metadata(self, objects_metadata,master_image_id):
        metadata_file = os.path.join(self.metadata_dir, 'image_objects_metadata.json')
        if os.path.exists(metadata_file):
            data = read_yaml(metadata_file)
        else:
            data = {"master_image": []}

        image_data = {"id": master_image_id, "objects": []}
        
        image_data["objects"].append(objects_metadata)

        data["master_image"].append(image_data)
        
        write_yaml(metadata_file, data)

        return metadata_file
