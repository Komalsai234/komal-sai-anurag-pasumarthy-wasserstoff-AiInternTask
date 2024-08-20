import easyocr
import os
import json
from project_root.utils.preprocessing import read_yaml,write_yaml

class TextExtractionModel:
    def __init__(self,master_image_id):
        """
        Initializes the text extraction model using EasyOCR.
        """
        self.reader = easyocr.Reader(['en'])
        self.master_image_id = master_image_id
        self.segmented_object_dir = f"project_root/data/segmented_objects/{self.master_image_id}"

    def extract_text(self, image_path):
        """
        Extracts text from a given image using OCR.
        Args:
            image_path (str): Path to the image file from which to extract text.
        Returns:
            extracted_text (str): The text extracted from the image.
        """
        results = self.reader.readtext(image_path)
        extracted_text = " ".join([result[1] for result in results])  
        return extracted_text.strip() 
    
    def extract_save_objects_text(self,metadata_file):

        try:
            if os.path.exists(metadata_file):
                metadata = read_yaml(metadata_file)
        except:
            raise Exception("Metadata file not found")

        # Iterate through the segmented object images
        for object_file in os.listdir(self.segmented_object_dir):
            object_id = object_file.split(".")[0]
            image_path = os.path.join(self.segmented_object_dir, object_file)
            print("Processing file:", image_path)

            # Extract text from the image
            extracted_text = self.extract_text(image_path)
            print("Extracted Text:", extracted_text)

            for master_image in metadata["master_image"]:
                if master_image["id"] == self.master_image_id: 
                    for objects in master_image["objects"]:
                        for obj in objects["objects"]:
                            if obj["object_id"] == object_id:  
                                obj["extracted_text"] = extracted_text  
                    break  



        write_yaml(metadata_file, metadata)