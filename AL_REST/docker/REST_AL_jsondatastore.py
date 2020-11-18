# original verison of REST_AL that uses json datastore
from flask import Flask, request
import json
import os.path
import pandas as pd
import urllib
import time
from flask_cors import CORS
import random

import efficient_annotation.datastores
from efficient_annotation.queuing import QueueManager
from efficient_annotation.common import load_config_file, get_annotation_completion, post_json, get_json
from efficient_annotation.common import Document, Annotation, JSONSerializable, AnnotationGroup
from efficient_annotation.statistics import StatisticsCalculator

app = Flask(__name__)
CORS(app)

# ############################
DEBUG = False
# ############################


config = load_config_file()

# load model definitions from config file
with open("/app/annotation_types.json", "r", encoding="utf-8") as fp:
    annotation_types = json.load(fp)
    model_types = {}
    for ann_type, data in annotation_types.items():
        for model_id in data["model_ids"]:
            model_types[model_id] = ann_type
   
ds = efficient_annotation.datastores.JSONDatastore(annotation_types, config['collections'])


SEGMENTATION_TOOL_DOCUMENT_ANNOTATION_PATH = config["segmentation-tool"]["path"] + config["segmentation-tool"]["document_annotation"]
SEGMENTATION_TOOL_GET_DOCUMENT_PATH = config["segmentation-tool"]["path"] + config["segmentation-tool"]["get_document"]

DOCUMENT_OVERVIEW_CSV = config["document_id_file"]["path"]
DOCUMENT_OVERVIEW_CSV_ENCODING = config["document_id_file"]["encoding"]
DOCUMENT_OVERVIEW_CSV_ENCODING_DELIMITER = config["document_id_file"]["delimiter"]
necessary_types = config["necessary_types"]

########################################################################
###### GLOBAL VARS
########################################################################
queue_manager = QueueManager(ds, annotation_types)

ds.initialize_scoring()


# statistics thread
statistics_calculator = StatisticsCalculator(ds, config, annotation_types)

########################################################################
###### INCOMING ROUTES
########################################################################
# 1. iterate through files in Document_Overview.csv -> "give me document annotation for document with id (document_id)
#	-> call url "http://segementation-tool.krex.svc:8080/document_annotation/" (POST with {"document_id": "example.pdf" })
# 2. upload (between AP4 and AP3)
# 3. /add_document/


#####
# 1.
#####
@app.route('/start_pipeline/', methods=['POST', 'GET'])
def start_pipeline():
    print("received "+request.method+" request: start_pipeline ")

    if request.method == 'POST':
        print("try to read from: " + DOCUMENT_OVERVIEW_CSV)

        df = pd.read_csv(DOCUMENT_OVERVIEW_CSV, sep=DOCUMENT_OVERVIEW_CSV_ENCODING_DELIMITER, encoding=DOCUMENT_OVERVIEW_CSV_ENCODING)
        print("Reading documents from csv and creating empty document...")
        t = time.time()
        t2 = time.time()
        idx = 0
        interval_size = 1000
        for document_id, origin, importance_score, label_text in zip(df['FileID'], df['Origin'], df['Importance'], df['Class']):
            if idx % interval_size == (interval_size-1):
                time_consumed = time.time() - t2
                t2 = time.time()
                fps = interval_size / time_consumed
                if fps == 0:
                    fps = 1
                print("Files per second: {:.2f}".format(fps))
            idx += 1
            document = Document(document_id, annotation_types) #TODO: who knows the page number?
            document.origin = origin
            document.initial_importance = importance_score
            document.importance_score = importance_score
            
            ######
            # ADD DOCUMENT LABEL
            # create the annotation
            annotation = Annotation(annotation_id="0", annotator_id="folder_model", annotator_type="model")
            vec = Annotation.label2vector(label_text=label_text, annotation_type="document_label", annotation_types=annotation_types)
            # if document label is unknown
            if vec == None:
                annotation.is_other = label_text
                annotation.annotation_vector = [0] * len(annotation_types["document_label"]["labels"])
            else:
                annotation.annotation_vector = vec    
            
            document_label = AnnotationGroup(group_id="0", annotation_type="document_label")
            document_label.annotations.append(annotation)
            document.document_label = document_label
            
            # assign the document to a priority or non-priority collection
            if importance_score == 1:
                document.annotation_completion = "priority"
                ds.save_to_collection(document, "priority")
            else:
                document.annotation_completion = "other"
                ds.save_to_collection(document, "other")
            payload = {"document_id": document_id}
            """
            if DEBUG:
                if origin != "Kaggle":
                    print(origin, SEGMENTATION_TOOL_DOCUMENT_ANNOTATION_PATH+document_id)
                continue
            post_json(payload, SEGMENTATION_TOOL_DOCUMENT_ANNOTATION_PATH+document_id)
            for i in range(TRY_LIMIT):
                time.sleep(1)
                document = json.loads(get_json(SEGMENTATION_TOOL_GET_DOCUMENT_PATH+document_id))
                #print("Recieved document:", document)
                if "      message" in document:
                    continue
                assert document_id == document["document_id"]
                if "pages" not in document:
                    print("ERROR: Still using wrong data structure! Please check the jupyter-notebook!")
                    break
                ds.update_document(document)
                break
            """
        print("Empty document creation complete", time.time() -t)
    return "Done!"

#####
# 2.
#####
"""
machine learning
--> See AP4
"""

#####
# 3.
#####
# --> see add_document

@app.route('/update_annotation_session_statistics/', methods=['POST'])
def update_annotation_session_statistics():
    return statistics_calculator.update_statistics()

@app.route('/get_annotation_session_statistics/')
def get_annotation_session_statistics():
    statistics_calculator.update_statistics()
    current_statistics = statistics_calculator.get_statistics()
    queue_manager.update_statistics(current_statistics)
    return current_statistics

get_annotation_session_statistics()

@app.route('/get_document_importance/')
def get_document_importance():
    document_list = ds.get_document_importance_from_datastore("partial")
    document_list = sorted(document_list, key=lambda document:document["importance_score"], reverse=True)
    return json.dumps([[document["document_id"], document["importance_score"]] for document in document_list])

@app.route('/next_most_important/<path:annotator_id>')
def next_most_important(annotator_id):
    # on its first call, sorts documents by importance_score
    # and then returns the most important
    # on all calls thereafter, returns the next most important
    next_document = queue_manager.get_next(annotator_id)
    return json.dumps(next_document)

@app.route('/init_next_most_important/', methods=['POST'])
def init_next_most_important_empty():
    print("Called init_next_most_important with no queue_type")
    print("Setting queue_type to default queue_type")
    return init_next_most_important("initial")

@app.route('/init_next_most_important/<path:queue_type>',  methods=['POST'])
def init_next_most_important(queue_type):
    # initializes the generator for next_most_important
    # also recalculates the document importance_score
    # supported queue_types:
    # 'standard': default queue_type; returns documents based on 'importance_score', ignoring documents that already have all 'necessary_types' completed by humans
    # 'initial': returns documents based on 'initial_importance', ignoring documents that already have annotations for 'document_contains' in 'annotated_by'
    # 'complete': returns documents that have been completed
    #get_annotation_session_statistics()
    return queue_manager.initialize_queue(queue_type)

@app.route('/get_document/<path:document_id>')
def get_document(document_id):
    document = ds.load_from_datastore(document_id)
    return json.dumps(JSONSerializable.serialize(document))

@app.route('/get_annotation_types/')
def get_annotation_types():
    return json.dumps(annotation_types)

@app.route('/get_model_types/')
def get_model_types():
    return json.dumps(ds.model_types)

@app.route('/add_document/', methods=['POST','GET'])
def add_document():
    error = None
    return_msg = "Please use POST request"
    if request.method == 'POST':
        json_document = request.get_json()
        document = Document.deserialize(json_document)
        if "target_collection" in json_document:
            target_collection = json_document["target_collection"]
        else:
            target_collection = get_annotation_completion(json_document, necessary_types)
            document.annotation_completion = target_collection
        ds.save_to_collection(document, target_collection)
        return_msg = target_collection
    return return_msg

@app.route('/get_current_queue/')
def get_current_queue():
    return json.dumps(queue_manager.get_current_queue())

@app.route('/update_timestamps/')
def update_timestamps():
    return_msg = "Finished update_timestamps to 2020-04-21"
    ds.update_timestamps()
    return return_msg

@app.route('/test/')
def test():
    return "test"

#app.run(host="0.0.0.0", port="5000")


