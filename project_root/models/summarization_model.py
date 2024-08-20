from transformers import pipeline
import os
import json
from project_root.utils.preprocessing import read_yaml,write_yaml

class SummarizationModel:
    def __init__(self, model_name="t5-small"):
        """
        Initializes the summarization model using a pre-trained T5 model.
        Args:
            model_name (str): Name of the pre-trained summarization model to use.
        """
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize_text(self, object_description):
        """
        Summarizes the given text.
        Args:
            object_metadata (str): The input text to summarize.
        Returns:
            summary (str): The summarized version of the input text.
        """

        if object_description['text']:
            description = f"Label: {object_description['label']}, text: {object_description['text']}"
            summary = self.summarizer(description, max_length=50, min_length=25, do_sample=False)
            return summary[0]['summary_text']
        else:
            return "" 
    
    def summarized_save_text(self, metadata_file_dir,master_image_id):
        """
        Saves the summarized text to a file.
        Args:
            metadata_file_dir (str): Path to the metadata JSON file.
        """

        try:
            if os.path.exists(metadata_file_dir):
                metadata = read_yaml(metadata_file_dir)
        except:
            raise Exception("Metadata file not found")

        for master_image in metadata["master_image"]:
            if master_image["id"] == master_image_id:  
                for obj_list in master_image["objects"]:  
                    for obj in obj_list["objects"]:  
                        object_description = {
                            'label': obj.get('label', 'Unknown'),  
                            'text': obj.get('extracted_text', '') 
                        }
                        obj["summary"] = self.summarize_text(object_description) 
                break  



        write_yaml(metadata_file_dir, metadata)