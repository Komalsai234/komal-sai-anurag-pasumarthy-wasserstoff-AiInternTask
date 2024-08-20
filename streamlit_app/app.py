import streamlit as st
import numpy as np
from PIL import Image
import os,sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from project_root.models.segmentation_model import SegmentationModel
from project_root.models.identification_model import IdentificationModel
from project_root.models.text_extraction_model import TextExtractionModel
from project_root.models.summarization_model import SummarizationModel
from project_root.utils.visualization import visualize_segmented_objects, display_segmented_object
from project_root.utils.preprocessing import save_input_image,generate_unique_id
from project_root.utils.postprocessing import save_artifacts

st.title("Object Segmentation")

upload_file = st.file_uploader("Upload the Image", type=['jpg', 'png', 'jpeg'])

generate_pred = st.button("Analyze")

if generate_pred:
    if upload_file is not None:
        image = Image.open(upload_file).convert('RGB')
        
        master_image_id = generate_unique_id()

        save_input_image(image,master_image_id)

        segmentation_model = SegmentationModel()
        results,id_list = segmentation_model.segment_image(image)

        
        identification_model = IdentificationModel()
        image_object = identification_model.identify_objects(results,id_list)
        metadata = identification_model.segmented_object_metadata(image_object, image,master_image_id)
        metadata_file_path = identification_model.save_metadata(metadata,master_image_id)

        text_extraction_model = TextExtractionModel(master_image_id)
        text_extraction_model.extract_save_objects_text(metadata_file_path)

        summarization_model = SummarizationModel()
        summarization_model.summarized_save_text(metadata_file_path,master_image_id)

        segmented_image_path = visualize_segmented_objects(image,metadata_file_path,master_image_id)
        st.subheader("Segmented Image:")
        st.image(segmented_image_path)


        st.subheader("")

        st.subheader("Segmented Objects:")

        display_segmented_object(master_image_id)

        st.subheader("")

        st.subheader("Text Extracted and Summarization:")

        df = save_artifacts(master_image_id,metadata_file_path)
        df.drop('Object ID',axis=1,inplace=True)

        st.dataframe(df)
