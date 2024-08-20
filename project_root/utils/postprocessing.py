import pandas as pd
import matplotlib.pyplot as plt
import os
import json

def save_artifacts(master_input_id,metadata_file_path,output_dir="project_root\data\output",):

    with open(metadata_file_path, 'r') as file:
        metadata = json.load(file)

    # Create a DataFrame to summarize the data
    object_data = []

    for master_image in metadata["master_image"]:
        if master_image["id"] == master_input_id:  
            for objects in master_image["objects"]:
                for obj in objects["objects"]:
                    object_data.append({
                        "Object ID": obj["object_id"],
                        "Label": obj["label"],
                        "Confidence": obj["confidence"],
                        "Bounding Box": obj["bbox"],
                        "Extracted Text": obj["extracted_text"],  
                        "Summary": obj["summary"] 
                    })
            break 

    df = pd.DataFrame(object_data)

    # Display the table
    print(df)

    # Save the table to a CSV file
    table_output_path = os.path.join(output_dir, f"{metadata['master_image'][0]['id']}", "output_summary.csv")
    os.makedirs(os.path.dirname(table_output_path), exist_ok=True)
    df.to_csv(table_output_path, index=False)

    return df