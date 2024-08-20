import sys
import os

# Add the parent directory of the current script to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from project_root.models.segmentation_model import SegmentationModel
import unittest
from PIL import Image

class TestSegmentationModel(unittest.TestCase):

    def setUp(self):
        self.model = SegmentationModel()
        self.test_image_path = "D:\\komal-sai-anurag-pasumarthy-wasserstoff-AiInternTask\\project_root\\data\\input_images\\000000039769.jpg"  # Provide the path to a test image

    def test_segment_image(self):
        input_image = Image.open(self.test_image_path)
        results, label_ids_list = self.model.segment_image(input_image)
        
        # Check if the results contain expected keys
        self.assertIn("scores", results)
        self.assertIn("labels", results)
        self.assertIn("boxes", results)
        
        # Check if label_ids_list is a dictionary
        self.assertIsInstance(label_ids_list, dict)

if __name__ == "__main__":
    unittest.main()
