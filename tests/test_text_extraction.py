import unittest

import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from project_root.models.text_extraction_model import TextExtractionModel

import os

class TestTextExtractionModel(unittest.TestCase):

    def setUp(self):
        # Use a relative path for the test image
        self.test_image_path = "D:\\komal-sai-anurag-pasumarthy-wasserstoff-AiInternTask\\project_root\\data\\input_images\\000000039769.jpg"
        self.model = TextExtractionModel()

    def test_extract_text(self):
        # Check if the extraction function returns text
        text = self.model.extract_text(self.test_image_path)
        self.assertIsInstance(text, str)

if __name__ == '__main__':
    unittest.main()
