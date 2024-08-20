import unittest
import os,sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from project_root.models.summarization_model import SummarizationModel

class TestSummarization(unittest.TestCase):

    def __init__(self) -> None:
        self.model = SummarizationModel()
    def setUp(self):
        # Mock object data for testing
        self.object_data = {
            'label': 'cat',
            'confidence': 0.99,
            'text': 'This is a test text extracted from the object.'
        }

    def test_summarization_output(self):
        # Test if the summarization function returns a summary string
        summary = self.model.summarize_text(self.object_data)
        self.assertIsInstance(summary, str)

    def test_summary_content(self):
        # Test if the summary contains key information from the object data
        summary = self.model.summarize_text(self.object_data)
        self.assertIn('cat', summary)
        self.assertIn('confidence', summary)

if __name__ == '__main__':
    unittest.main()
