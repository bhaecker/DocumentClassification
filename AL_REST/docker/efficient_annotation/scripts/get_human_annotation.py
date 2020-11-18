import json
import os.path
from common import Document, get_json


# get human annotations from document 
def get_human_annotation():

        path = os.path.join(r"C:\Users\anna_\OneDrive\Documents\ActiveLearning\AL_REST\test_data\complete\000dfd749a544371af076f6d80a448db.json")
        with open(path, "r") as fp:

            # --- load document json from file ---
            document = json.load(fp) 

            # --- or get a document json from db ---
            # base_url = "http://localhost:5000"
            # document = json.loads(get_json(base_url+"/get_document/000dfd749a544371af076f6d80a448db"))

            # convert to Document object
            document_obj = Document.deserialize(document) 
            for annotation in document_obj.annotations():
                print("")
                print("annotation_id " + annotation.annotation_id)
                print("annotator_id "+ annotation.annotator_id)
                print("annotator_type " + annotation.annotator_type)
                print("annotation_vector " + str(annotation.annotation_vector))
                print("annotation_type " + annotation.annotation_type)


get_human_annotation()