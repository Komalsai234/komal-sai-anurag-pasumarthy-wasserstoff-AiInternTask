import unittest
from PIL import Image
import sys
import os
import torch  # Import torch for tensor handling

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from project_root.models.identification_model import IdentificationModel

class TestIdentificationModel(unittest.TestCase):

    def setUp(self):
        self.model = IdentificationModel()

    def test_identify_objects(self):
        results = {
            'scores': torch.tensor([0.9982, 0.9960, 0.9955, 0.9988, 0.9987]),
            'labels': torch.tensor([75, 75, 63, 17, 17]),
            'boxes': torch.tensor([
                [40.163, 70.812, 175.55, 117.98],
                [333.24, 72.550, 368.33, 187.66],
                [-2.2602, 1.1496, 639.73, 473.76],
                [13.241, 52.055, 314.02, 470.93],
                [345.40, 23.854, 640.37, 368.72]
            ])
        }
        label_ids_list = {75: "Test Label", 63: "Another Label", 17: "Different Label"}

        objects = self.model.identify_objects(results, label_ids_list)
        
        # Check if the objects list is not empty and has correct data
        self.assertGreater(len(objects), 0)
        self.assertIn("Test Label", [obj["label"] for obj in objects])
        self.assertTrue(any(obj["confidence"] >= 0.95 for obj in objects))

    def test_segmented_object_metadata(self):
        # Create a blank image
        image = Image.new('RGB', (100, 100), color='white')

        objects = [{
            "label": "Test Label",
            "confidence": 0.95,
            "bbox": [10, 20, 30, 40]
        }]
        
        metadata = self.model.segmented_object_metadata(objects, image)
        
        self.assertIn('id', metadata)
        self.assertIn('objects', metadata)
        self.assertEqual(len(metadata['objects']), 1)
        self.assertEqual(metadata['objects'][0]['label'], "Test Label")
        self.assertEqual(metadata['objects'][0]['confidence'], 0.95)
        self.assertEqual(metadata['objects'][0]['bbox'], [10, 20, 30, 40])

if __name__ == '__main__':
    unittest.main()
