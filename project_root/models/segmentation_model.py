import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image

class SegmentationModel:
    def __init__(self, model_name="facebook/detr-resnet-50"):
        """
        Initializes the segmentation model using DETR (DEtection TRansformers).

        Args:
            model_name (str): Name of the pre-trained model to use.

        """
        self.processor = DetrImageProcessor.from_pretrained(model_name, revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained(model_name, revision="no_timm")

    def segment_image(self, image):
        """
        Performs object detection and segmentation on the input image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            results (dict): Dictionary containing detected objects and their attributes.
            image (PIL.Image): The input image in PIL format.

        """
        inputs = self.processor(images=image, return_tensors="pt")  
        outputs = self.model(**inputs)  

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        label_ids_list = self.model.config.id2label

        return results,label_ids_list
