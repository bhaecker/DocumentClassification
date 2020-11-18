# uses mongodb as datastore
from flask import Flask, request
import json
import os.path
import pandas as pd
import urllib
import time
from flask_cors import CORS
import random
import copy
from shutil import copyfile
from math import floor
import threading
import traceback


import efficient_annotation.datastores
from efficient_annotation.queuing import QueueManager
from efficient_annotation.common import load_config_file, get_annotation_completion, post_json, get_json
from efficient_annotation.common import Document, Annotation, JSONSerializable, AnnotationGroup
from efficient_annotation.statistics import StatisticsCalculator
from efficient_annotation.datastores import MongoDB
from efficient_annotation.logging import Logger


app = Flask(__name__)
CORS(app)

logger = Logger()
config = load_config_file()
lock = threading.Lock()

# ############################
#DEBUG = False
VERBOSE = True

DEBUG_extend_annotation_vectors = config["debug"]["DEBUG_extend_annotation_vectors"] # if False: use filtered annotation types
DEBUG_prediction = config["debug"]["DEBUG_prediction"] # if False: use public/Document_Overview.csv otherwise Document_Overview_sm.csv
DEBUG_evalutation = config["debug"]["DEBUG_evalutation"] # if False: send requests to AP4
DEBUG_evalutation_annotation_types = config["debug"]["DEBUG_evalutation_annotation_types"] # if False: use filtered annotation types
#DEBUG_remove_documents_if_less_than_X_examples = config["debug"]["DEBUG_remove_documents_if_less_than_X_examples"]

WARN_about_annotation_version = True
WARN_about_has_base = True
 
# ############################


# load model definitions from config file
with open("/app/annotation_types.json", "r", encoding="utf-8") as fp:
    annotation_types = json.load(fp)
    model_types = {}
    for ann_type, data in annotation_types.items():
        for model_id in data["model_ids"]:
            model_types[model_id] = ann_type
   
# load finetuned_annotation_types.json 
# this file will be filled out manually during the workshop
with open("/app/finetuned_annotation_types.json", "r", encoding="utf-8") as fp:
    finetuned_annotation_types = json.load(fp)
    finetuned_model_types = {}
    for ann_type, data in finetuned_annotation_types.items():
        for model_id in data["model_ids"]:
            finetuned_model_types[model_id] = ann_type  


ds = MongoDB(annotation_types, config['collections'], config['database']['url'])

SEGMENTATION_TOOL_DOCUMENT_PATH = config["segmentation-tool"]["path"]
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

#ds.initialize_scoring()

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

# starts predictions at AP4
@app.route('/start_prediction/', methods=['POST'])
def start_prediction():

    logger.log("start_prediction")

    ### read Overview.csv ###
    if DEBUG_prediction == True:
        df = pd.read_csv(r"Document_Overview_sm.csv", sep=DOCUMENT_OVERVIEW_CSV_ENCODING_DELIMITER, encoding=DOCUMENT_OVERVIEW_CSV_ENCODING)
    else:
        df = pd.read_csv(DOCUMENT_OVERVIEW_CSV, sep=DOCUMENT_OVERVIEW_CSV_ENCODING_DELIMITER, encoding=DOCUMENT_OVERVIEW_CSV_ENCODING)

    max_batch_size = config["max_batch_size"]

    t_start = time.time()
    document_id_list = []

    ### iterate over files in Overview.csv and collect all doc_ids###
    for document_id in df['FileID']:

        result = get_document(document_id)

        # append if doc not in db
        if result == "{}": 
            document_id_list.append(document_id)
        # append if in db and has_base == False
        elif (result != "{}" and "has_base" in result and result["has_base"] == False):
            document_id_list.append(document_id)
        else:
            logger.log("SKIP id " + str(document_id) + " do not send to predict_batch_with_base_model.")
            if not "has_base" in result:
                logger.log("WARNING: id " + str(document_id) + " does not have field has_base.")


    path = SEGMENTATION_TOOL_DOCUMENT_PATH +"predict_batch_with_base_model/"    
    ### send request to AP4 to start prediction on document with document_id ###
    send_post_requests_with_max_batchsize(document_id_list, path)

    t_end = time.time()

    message = ""

    if (len(document_id_list) > 0):
        message = ("start_prediction done. Sent " + str(len(document_id_list)) 
        + " ids to "+ SEGMENTATION_TOOL_DOCUMENT_PATH +"predict_batch_with_base_model"
        + "; total time elapsed (for sending all requests): " + str(t_end-t_start) 
        + "; avg. time per doc: " + str((t_end-t_start) / len(document_id_list)))
    else:
        message = "### WARNING DOCUMENT_OVERVIEW_CSV contains 0 document_ids ###"

    logger.log(message)

    return message
        

def send_post_requests_with_max_batchsize(dataset, path):
    max_batch_size = config["max_batch_size"]

    logger.log("start: send_post_requests_with_max_batchsize to path " + path)
    logger.log("max_batchsize: " + str(max_batch_size))
    logger.log("total length of dataset: " + str(len(dataset)))
    logger.log("complete dataset: " + str(dataset))


    # # remove logged ids from dataset
    # if os.path.exists(logger.get_path_to_logged_file("ids_logged_during_prediction.txt")):
    #     dataset_copy = copy.deepcopy(dataset)
    #     with open(logger.get_path_to_logged_file("ids_logged_during_prediction.txt")) as fp:
    #         line = fp.readline()
    #         while line:
    #             line = line.strip()
    #             if line in dataset:
    #                 dataset_copy.remove(line)
    #                 logger.log("SKIP id " + line + " in call to " + path)
    #             line = fp.readline()

    #     dataset = dataset_copy

    if max_batch_size == 0:
        error_msg = "ERROR. max_batch_size must be > 0"
        logger.log(error_msg)
        return error_msg

    if len(dataset) == 0:
        error_msg = "WARNING. send_post_requests_with_max_batchsize. dataset is empty. no request sent to " + path
        logger.log(error_msg)
        return error_msg

    counter = 0

    document_id_list = []

    # send batches of size max_batch_size 
    for item in dataset:
        document_id_list.append(item)

        if len(document_id_list) == max_batch_size:
            counter += len(document_id_list)
            send_one_post_request_with_max_batchsize(document_id_list, path)            
            document_id_list = []
                    
    # send rest of documents (batch of size smaller than max_batch_size)
    if len(document_id_list) > 0:

        counter += len(document_id_list)
        send_one_post_request_with_max_batchsize(document_id_list, path)

    # remove all logged ids from file
    # logger.log_output_only("", "ids_logged_during_prediction.txt", "w")

    logger.log("CHECK. sent: "  + str(counter) + ". total length of dataset: " + str(len(dataset)))
    logger.log("Done. send_post_requests_with_max_batchsize to path " + path)        

    return "Done. send_post_request_with_max_batchsize"


# called by send_post_requests_with_max_batchsize
def send_one_post_request_with_max_batchsize(document_id_list, path):
    
    logger.log("call " + path)
    logger.log("with " + str(len(document_id_list)) + " document ids.")
    logger.log("payload: " + str(document_id_list))

    repeated_post_json(document_id_list, path)

    # log sent ids to file - will not be logged if the request does not return
    #for doc_id in document_id_list:
    #    logger.log_output_only(str(doc_id)+"\n", "ids_logged_during_prediction.txt", "a")
    return "Done. send_one_post_request_with_max_batchsize"


# if post fails - repeat posting json 
# if post works - do not send repeatedly
def repeated_post_json(payload, path):
    total_wait_time = 0
    keep_trying = True

    try:
        post_json(payload, path)
    except:
        while keep_trying:
            try:
                logger.log("ERROR: could not reach "+str(path)+ " try again")
                post_json(payload, path)
                keep_trying = False 
            except:
                if DEBUG_prediction == True:
                    waittime = 1
                    logger.log("sleep "+str(waittime)+" sec. total_wait_time = " + str(total_wait_time))
                    time.sleep(waittime)
                    total_wait_time += waittime

                    if total_wait_time > 1:
                        keep_trying = False

                else:
                    waittime = config["pause_secs_after_failed_request"]
                    logger.log("sleep "+str(waittime)+" sec. total_wait_time = " + str(total_wait_time))
                    time.sleep(waittime)            
                    total_wait_time += waittime



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
    return ds.get_document_importance_from_datastore("partial") # TODO why partial only

@app.route('/next_most_important/<path:annotator_id>')
def next_most_important(annotator_id):
    # on its first call, sorts documents by importance_score
    # and then returns the most important
    # on all calls thereafter, returns the next most important
    with lock:
        next_document = "{}"

        # an error will occur if next_most_important is called before
        # init_next_most_important
        try:
            next_document = queue_manager.get_next(annotator_id)
        except Exception as e:
            logger.log("ERROR in REST_AL: Could not get next_most_important. Please call init_next_most_important before next_most_important")
            logger.log("ERROR" + str(type(e)))    
            logger.log(e.args)     
            logger.log(e)
            traceback.print_exc()   
            pass
        return json.dumps(next_document)

@app.route('/init_next_most_important/', methods=['POST'])
def init_next_most_important_empty():
    print("Called init_next_most_important with no queue_type")
    print("Setting queue_type to default queue_type")
    return init_next_most_important("initial")

@app.route('/init_next_most_important/<path:queue_type>',  methods=['POST'])
def init_next_most_important(queue_type):

    print("REST_AL: init_next_most_important with queue_type " + queue_type)

    # initializes the generator for next_most_important
    # also recalculates the document importance_score
    # supported queue_types:
    # 'standard': default queue_type; returns documents based on 'importance_score', ignoring documents that already have all 'necessary_types' completed by humans
    # 'initial': returns documents based on 'initial_importance', ignoring documents that already have annotations for 'document_contains' in 'annotated_by'
    # 'complete': returns documents that have been completed
    #get_annotation_session_statistics()
    return queue_manager.initialize_queue(queue_type)


def update_scores_in_current_queue():
    return queue_manager.update_scores_in_current_queue()

@app.route('/get_document/<path:document_id>')
def get_document(document_id):
    #print("ds.load_from_datastore_as_json(document_id) " + str(ds.load_from_datastore_as_json(document_id)), flush = True)
    response = ds.load_from_datastore_as_json(document_id)
    if response != None:
        return response
    else:
        logger.log("WARNING: in get_document/document_id: no document with document_id "+ str(document_id)+ " was found. return {}")
        return json.dumps({})

@app.route('/get_all_documents/<path:collection>')
def get_all_documents_in_collection(collection):
    docs = ds.get_all_documents_in_collection(collection)
    return json.dumps(list(docs))


def get_documents_by_query_as_cursor(query):
    docs = ds.get_documents_by_query(query)
    return docs



# returns the total number of documents in the database
@app.route('/get_number_of_documents')
def get_number_of_documents_all_docs():
    result = {"num" : ds.get_number_of_documents()}
    return json.dumps(result)

@app.route('/get_number_of_documents_for_all_collections')
def get_number_of_documents_for_all_collections():
    result = ds.get_num_docs_per_target_collection()
    return json.dumps(result)


@app.route('/get_number_of_documents/<path:collection>')
def get_number_of_documents(collection):
    result = {"num" : ds.get_number_of_documents(collection)}
    return json.dumps(result)

@app.route('/get_all_documents/')
def get_all_documents():
    docs = ds.get_all_documents()
    # turn cursor into json list
    return json.dumps(list(docs))

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

        if VERBOSE:
            logger.log("add/update document " + str(json_document["document_id"]))    

        # fill up annotation vectors with zeros if document_label comes from finetuned model
        json_document = extend_annotation_vectors(json_document)

        ds.save_to_collection(json_document)

        return_msg = "saved document"
    return return_msg



# fill up document_label annotation vector with zeros 
# if document_label prediction comes from finetuned model
# ==> to find out we check if annotation.annotator_type is "model_finetuned"
# extended annotation_vectors will have same length 
# as finetuned_annotation_types["document_label"]["labels"]
# TODO later: also for other annotation vectors (now only document_label)
def extend_annotation_vectors(json_document):

    for annotation in json_document["document_label"]["annotations"]:
        if annotation["annotator_type"] == "model_finetuned": 

            finetuned_annotation_types = json.loads(get_finetuned_annotation_types())
            filtered_finetuned_annotation_types = json.loads(get_filtered_finetuned_annotation_types())

            #print(finetuned_annotation_types, flush = True)

            finetuned_document_labels = finetuned_annotation_types["document_label"]["labels"]
            filtered_document_labels = filtered_finetuned_annotation_types["document_label"]["labels"]

            if DEBUG_extend_annotation_vectors:
                filtered_document_labels = json.loads(get_annotation_types())["document_label"]["labels"]
                logger.log("##### WARNING DEBUG extend_annotation_vectors -- should be turned off when deployed #####")
                annotation["annotation_vector"] = [-1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
                filtered_document_labels.pop(8)
                filtered_document_labels.pop(0)
                annotation["annotation_vector"].pop(8)
                annotation["annotation_vector"].pop(0)

            annotation_vector_short = annotation["annotation_vector"]
            annotation_vector_long = [0] * len(finetuned_document_labels)

            ptr_filtered = 0
            for ptr_finetuned in range(len(finetuned_document_labels)):
                #print("annotation_vector_long",annotation_vector_long)
                #print("annotation_vector_short",annotation_vector_short)
                #print("ptr_filtered",ptr_filtered)
                #print("ptr_finetuned",ptr_finetuned, flush = True)
                
                if ptr_filtered == len(filtered_document_labels) or finetuned_document_labels[ptr_finetuned] != filtered_document_labels[ptr_filtered]: 
                    annotation_vector_long[ptr_finetuned] = 0
                else: 
                    annotation_vector_long[ptr_finetuned] = annotation_vector_short[ptr_filtered] 
                    ptr_filtered += 1


            #logger.log("annotation_vector_long" + str(annotation_vector_long))
            #logger.log("annotation_vector_short" + str(annotation_vector_short))
            
            annotation["annotation_vector"] = annotation_vector_long 

    return json_document


# store all documents in a folder to database
# pass path to folder as parameter in rest call (json files can be in top level of folder or inside subfolders)
# surround path with single quotes (') if path contains slashes  
# Example REST call:
# /add_documents_in_folder/'copy_files_to_db/test_files'
@app.route('/add_documents_in_folder/<path:path_to_folder>', methods=['POST'])
def add_documents_in_folder(path_to_folder):
    error = None
    return_msg = "Please use POST request"
    if request.method == 'POST':
        ds.save_all_documents_in_folder(path_to_folder)
        return_msg = "saved documents"
    return return_msg


# for testing only!
@app.route('/empty_db/', methods=['POST'])
def empty_db():
    ds.empty_db()
    queue_manager.reset_partial_queue_counter()
    return "Emptied Database."


@app.route('/empty_log/<path:filename>', methods=['POST'])
def empty_log(filename):
    logger.empty_log(filename)
    return "Emptied log file " + str(filename)


# for testing only
# this adds an empty annotation by annotator to document with document_id
# used to test multi annotation queue
#@app.route('/add_annotation/<path:document_id>/<path:annotator_id>/', methods=['POST'])
#def add_annotation(document_id, annotator_id):
#    ds.add_annotation(document_id, annotator_id)
#    return "Added annotation."

@app.route('/update_timestamps/')
def update_timestamps():
    return_msg = "Finished update_timestamps to 2020-04-21"
    ds.update_timestamps()
    return return_msg

@app.route('/test/')
def test():
    return "test"


#### REST calls used for evaluation
# they are called by AP2/GUI
# calls go to AP4, AP4 then calls AP3
# response from AP3 with add_document and train_done
# see docker/AP2-AP3-AP4-communication-V3.pdf for details


current_evalutation_scenario = "" 
test_set = []
initial_finetuning_batch = []
al_finetuning_batches = []
random_finetuning_batches = []
all_except_seen = []
all_except_seen_copy = []


if DEBUG_evalutation:
    logger.log("##### WARNING: evaluation is in DEBUG mode: does not call AP4 rest calls (if deployed: should be DEBUG_evalutation = False) #####")
else:
    logger.log("##### evalutation rest calls NOT in DEBUG mode: calls AP4 at "+ SEGMENTATION_TOOL_DOCUMENT_PATH +" #####")

if DEBUG_evalutation_annotation_types:
    logger.log("##### WARNING: evaluation is in DEBUG mode: uses old (not fine-tuned) annotation_types #####")

# runs entire evaluation pipeline with default parameters
@app.route('/start_evaluation/', methods=['POST'])
def start_evaluation_no_params():
    logger.log("split_dataset with default paramters: " 
    + "min_finetune_documents " + str(config["evaluation"]["min_finetune_documents"] ) 
    + "; min_test_documents " + str(config["evaluation"]["min_test_documents"])
    + "; min_al_documents" + str(config["evaluation"]["min_al_documents"]))

    start_evaluation(config["evaluation"]["min_finetune_documents"], config["evaluation"]["min_test_documents"], config["evaluation"]["min_al_documents"])
    logger.log("Done. Start_evaluation with default parameters.")
    return("Done. Start_evaluation with default parameters.")

# runs entire evaluation pipeline
@app.route('/start_evaluation/<path:min_finetune_documents>/<path:min_test_documents>/<path:min_al_documents>', methods=['POST'])
def start_evaluation(min_finetune_documents,min_test_documents,min_al_documents):
    split_dataset(min_finetune_documents,min_test_documents,min_al_documents)
    start_base_predictions()
    start_finetuning_training()
    start_active_learning("efficient_annotation")
    start_active_learning("random")
    logger.log("Finished evaluation.")
    return "Finished evaluation."

@app.route('/split_dataset/', methods=['POST'])
def split_dataset_no_params():
    logger.log("split_dataset with default paramters: " 
    + "min_finetune_documents " + str(config["evaluation"]["min_finetune_documents"] ) 
    + "; min_test_documents " + str(config["evaluation"]["min_test_documents"])
    + "; min_al_documents" + str(config["evaluation"]["min_al_documents"]))

    split_dataset(config["evaluation"]["min_finetune_documents"], config["evaluation"]["min_test_documents"], config["evaluation"]["min_al_documents"])
    logger.log("Done. split dataset")
    return("Done. Split dataset with default parameters.")


def compute_split_percentages(min_finetune_documents, min_test_documents, min_al_documents):
    
    try:
        min_finetune_documents = int(min_finetune_documents)
        min_test_documents = int(min_test_documents)
        min_al_documents = int(min_al_documents)
    except:
        error_msg = "ERROR: dataset split parameters must be integers"
        logger.log(error_msg)
        return error_msg
    
    total_min_documents = min_finetune_documents + min_test_documents + min_al_documents

    dataset_split_percentages = dict();  
    dataset_split_percentages['initial_finetune_split'] = min_finetune_documents / total_min_documents 
    dataset_split_percentages['test_split'] =  min_test_documents / total_min_documents
    dataset_split_percentages['al_split'] = min_al_documents / total_min_documents

    logger.log("min_finetune_documents " + str(min_finetune_documents) + "; min_test_documents " + str(min_test_documents) + "; min_al_documents" + str(min_al_documents))
    logger.log("dataset_split_percentages " + str(dataset_split_percentages))

    return dataset_split_percentages


# this call must be done at the beginning of evaluation
#@app.route('/split_dataset/<path:test_split>/<path:initial_finetune_split>', methods=['POST'])
@app.route('/split_dataset/<path:min_finetune_documents>/<path:min_test_documents>/<path:min_al_documents>', methods=['POST'])
def split_dataset(min_finetune_documents, min_test_documents, min_al_documents):
    logger.log("split_dataset with parameters: min_finetune_documents:" + str(min_finetune_documents) + " min_test_documents: " + str(min_test_documents) + " min_al_documents: " +min_al_documents)

    global current_evalutation_scenario
    global test_set
    global initial_finetuning_batch
    global al_finetuning_batches
    global random_finetuning_batches
    global all_except_seen
    global all_except_seen_copy

    ### re-set datasets and target collections, so that we get the same result if we run the evaluation pipeline multiple times #
    # reset datasets
    test_set = []
    initial_finetuning_batch = []
    al_finetuning_batches = []
    random_finetuning_batches = []

    all_except_seen = []
    all_except_seen_copy = []

    # reset target_collections
    for tag in ["_test_set", "_initial_finetune_set", "_other", "_not_enough_examples", "_unconfirmed"]:
        ds.set_target_collection_old_to_new("has_complete_human_annotation"+tag, "has_complete_human_annotation")
        ds.set_target_collection_old_to_new("has_partial_human_annotation"+tag, "has_partial_human_annotation")
    # end: reset

    try:
        min_finetune_documents = int(min_finetune_documents)
        min_test_documents = int(min_test_documents)
        min_al_documents = int(min_al_documents)
    except:
        error_msg = "ERROR: dataset split parameters must be integers"
        logger.log(error_msg)
        return error_msg

    dataset_split_percentages = compute_split_percentages(min_finetune_documents, min_test_documents, min_al_documents)
    test_split = dataset_split_percentages["test_split"]
    initial_finetune_split = dataset_split_percentages["initial_finetune_split"]
    total_min_documents = min_finetune_documents + min_test_documents + min_al_documents

    logger.log("\n------------------------------------- EVALUATION -------------------------------------------")
    logger.log("split dataset with test_split: " + str(test_split) + " initial_finetune_split: " + str(initial_finetune_split))

    # at this point: human annotation is finished, 
    # completely annotated documents will have
    # target_collection = "has_complete_human_annotation"

    # 1. from db get a list of doc_ids with human annotations (target_collection = "has_complete_human_annotation" & "has_partial_human_annotation") from database
    documents_complete = ds.get_all_document_ids_in_collection("has_complete_human_annotation") 
    documents_partial = ds.get_all_document_ids_in_collection("has_partial_human_annotation") 

    # 2. split data into:
    # test_set (used on base model and fine-tuned model)
    # fine_tune_train_batches -> created by active learning (used to train fine-tuned models)
    # all_except_seen (all documents except seen train data and test data)

    documents = list(documents_complete)
    for i, item in enumerate(documents):
        documents[i] = item["document_id"]

    logger.log("documents with complete annotation: " + str(documents))

    documents_partial = list(documents_partial)
    for i, item in enumerate(documents_partial):
        documents_partial[i] = item["document_id"]

    logger.log("documents with partial annotation: " + str(documents_partial))

    logger.log(str(documents_partial), "all_docs_partial.txt", "w")
    logger.log("len: " + str(len(documents_partial)), "all_docs_partial.txt", "a")
    
    logger.log(str(documents), "all_docs_complete.txt", "w")
    logger.log("len: " + str(len(documents)), "all_docs_complete.txt", "a")

    documents.extend(documents_partial) # documents inlcudes completely and partially annotated documents

    total_number_of_documents = len(documents)
    logger.log("number of documents in target_collections has_complete_human_annotation and has_partial_human_annotation before removing documents: " + str(total_number_of_documents))


    # count number of documents before excluding 
    counter_per_document_label = count_documents_per_documentlabel(documents)
    logger.log("number of documents per document label before filtering: " +str(counter_per_document_label))


    # ------------- exclude documents with annotation_vectors [0,0,0...] (zeros only) which
    # indicates that human created a new class --------------------------------------------
    documents = exclude_is_other(documents)
    logger.log("number of documents after excluding documents labeled as is_other: " + str(len(documents)))
    logger.log(str(count_documents_per_documentlabel(documents)))

    documents = exclude_not_confirmed(documents)
    logger.log("number of documents after removing missing/unconfirmed document label: " + str(len(documents)))
    logger.log(str(count_documents_per_documentlabel(documents)))

    # randomize documents
    random.seed(30)
    random.shuffle(documents)

    # ----------------------- filter document labels with number of examples < threshold -----------------------------
    logger.log("for evaluation: exclude documents if less than " + str(total_min_documents) + " per document label.")
    
    threshold = total_min_documents
    document_labels = filter_document_labels(documents, threshold)
    logger.log(str(count_documents_per_documentlabel(documents)))

    logger.log("filtered document labels: " + str(document_labels))
    number_of_document_labels = len(document_labels)
    logger.log("number of filtered document_labels: " + str(number_of_document_labels))

    if number_of_document_labels == 0:
        logger.log("ERROR: all document labels removed when filtering. In config.js, set DEBUG_remove_documents_if_less_than_X_examples to a lower value or annotate more documents.")
        return "ERROR: all document labels removed when filtering."


    # ----------------------- counter number of documents for each document_label ------------------
    counter_per_document_label = count_documents_per_documentlabel(documents)

    # ----------------------- remove docs with less than threshold examples ------------------
    logger.log("remove docs with < " + str(threshold) + " examples")
    documents = remove_documents_if_not_enough_examples(documents, counter_per_document_label, threshold)

    # set counter_per_document_label to 0 where count < threshold
    for idx, item_count in enumerate(counter_per_document_label):
        if item_count < threshold:
            counter_per_document_label[idx] = 0

    logger.log("number of documents per document label: " +str(counter_per_document_label))


    total_number_of_documents = len(documents)
    logger.log("number of documents after removing documents with zero-only human annotation and documents with < " + str(threshold)+ " examples: " + str(total_number_of_documents))

    logger.log(str(list(documents)), "all_docs_filtered.txt", "w")
    logger.log("len: " + str(len(documents)), "all_docs_filtered.txt", "a")

    # ----------------------- generate test set ------------------

    # put <test_split> percent of documents in test_set
    number_of_documents_in_test_set = floor(total_number_of_documents * test_split)
    print("number_of_documents_in_test_set " + str(number_of_documents_in_test_set))
    # get first <number_of_documents_in_test_set> items in document
    test_set = documents[0:number_of_documents_in_test_set]
    print("test set contains " + str(len(test_set)) + " items.")

    # for all in <test_set> change target_collection to "has_complete_human_annotation_test_set" 
    for doc_id in test_set:
        doc = ds.load_from_datastore_as_json(doc_id)
        doc["target_collection"] = doc["target_collection"]+"_test_set"
        ds.save_to_collection(doc)

    # put all except docs in test_set in all_except_seen
    all_except_seen = documents[number_of_documents_in_test_set:]
    print("all_except_seen contains " + str(len(all_except_seen)) + " items.", flush = True)

    # initial_fine_tuning: 
    # select batch for fine-tuning the base model.
    # this batch will be used to train the base model
    # which is the starting point for active learning

    # select a percentage of each class for fine-tuning
    # only document labels are finetuned

    # take <initial_finetune_split> percent of counter and that number of  documents to <initial_finetuning_batch>
    number_of_docs_in_initial_finetuned_batch = [0] * len(counter_per_document_label)
    
    for i, item in enumerate(counter_per_document_label):
        number_of_docs_in_initial_finetuned_batch[i] = floor(item * initial_finetune_split)

    logger.log("number_of_docs_in_initial_finetuned_batch: " + str(number_of_docs_in_initial_finetuned_batch))

    initial_finetuning_batch = []
    for doc_id in all_except_seen:
        human_annotations = get_human_annotations(doc_id)
        for annotation in human_annotations["annotations"]:            
            if annotation["annotation_type"] == "document_label":
                # check if counter for current document class > 0
                if 1 in annotation["annotation_vector"] and number_of_docs_in_initial_finetuned_batch[annotation["annotation_vector"].index(1)] > 0:
                    #print("add " + doc["document_id"] , flush = True)
                    initial_finetuning_batch.append(doc_id)
                    number_of_docs_in_initial_finetuned_batch = [x - y for x, y in zip(number_of_docs_in_initial_finetuned_batch, annotation["annotation_vector"])] # count down in number_of_docs_in_initial_finetuned_batch
                

    logger.log(str(initial_finetuning_batch),"finetuning_batch0.txt", "w")
    logger.log(str(len(initial_finetuning_batch)), "finetuning_batch0.txt", "a")

    print("all_except_seen - before removing initial_finetuning_batch")
    print(len(all_except_seen))

    # remove from all_except_seen
    for doc_id in initial_finetuning_batch:
        #print("remove", flush = True)
        #print(doc_id, flush = True)
        all_except_seen.remove(doc_id)
        doc = ds.load_from_datastore_as_json(doc_id)
        doc["target_collection"] = doc["target_collection"]+"_initial_finetune_set"
        ds.save_to_collection(doc)

    print("all_except_seen - after removing initial_finetuning_batch")
    print(len(all_except_seen))

    all_except_seen_copy = copy.deepcopy(all_except_seen)
    
    logger.log("first dataset split finished.")
    logger.log("initial_finetuning_batch: " + str(initial_finetuning_batch))

    logger.log("all_except_seen: " + str(all_except_seen))
    logger.log("test_set: " + str(test_set))

    logger.log("size of datasets: " + "test_set: " + str(len(test_set)) + " all_except_seen: "+ str(len(all_except_seen)) + " initial_finetuning_batch: " + str(len(initial_finetuning_batch)))
    logger.log("in total: " + str(len(test_set) + len(all_except_seen) + len(initial_finetuning_batch)) + "; total_number_of_documents: " + str(total_number_of_documents))

    logger.log(str(test_set), "test_set.txt", "w")
    logger.log(str(len(test_set)), "test_set.txt", "a")

    if len(initial_finetuning_batch) == 0: 
        logger.log("ERROR: initial_finetuning_batch is empty.")
        return "ERROR: initial_finetuning_batch is empty."

    return "First dataset split done: split into initial_finetuning_batch, test set and all_except_seen."


def filter_document_labels(documents, threshold):

    counter_per_document_label = count_documents_per_documentlabel(documents)
    
    # ------ filter out document labels that do not have enough examples -----
    # remove from documents and from finetuned_annotation_types

    # load filtered_finetuned_annotation_types.json
    if DEBUG_evalutation_annotation_types == True:
        finetuned_annotation_types = json.loads(get_annotation_types()) 
    else:
        finetuned_annotation_types = json.loads(get_finetuned_annotation_types())

    finetuned_documens_labels = finetuned_annotation_types["document_label"]["labels"]

    logger.log("finetuned document labels: " + str(finetuned_documens_labels))

    # remove document labels with number below threshold in counter_per_document_label 
    
    filtered_finetuned_documens_labels = copy.deepcopy(finetuned_documens_labels)
    for idx, count in enumerate(counter_per_document_label):
        if count < threshold: 
            filtered_finetuned_documens_labels.remove(finetuned_documens_labels[idx])

    finetuned_annotation_types ["document_label"]["labels"] = filtered_finetuned_documens_labels

    # store file filtered_finetuned_annotation_types.json

    logger.log_output_only(json.dumps(finetuned_annotation_types), "filtered_finetuned_annotation_types.json","w")

    return filtered_finetuned_documens_labels


def remove_documents_if_not_enough_examples(documents, counter_per_document_label, threshold):
    documents_copy = copy.deepcopy(documents)

    for doc_id in documents:
        human_annotations = get_human_annotations(doc_id)  
        for annotation in human_annotations["annotations"]:
            if annotation["annotation_type"] == "document_label":

                # get maximum of human annotation
                vec = annotation["annotation_vector"]
                m = max(vec)
                index = [i for i, j in enumerate(vec) if j == m]

                #check if there are less than <threshold> examples in total for that index (the document label of this example)
                if len(index) == 1 and counter_per_document_label[index[0]] < threshold: # TODO: check length of vectors matches
                    documents_copy.remove(doc_id)

                    # set target_collection
                    result = get_document(doc_id)
                    result["target_collection"] = result["target_collection"]+"_not_enough_examples"

                    ds.save_to_collection(result)

    return documents_copy


def count_documents_per_documentlabel(documents):
    if DEBUG_evalutation_annotation_types == True:
        annotation_types = json.loads(get_annotation_types())
        document_labels = annotation_types["document_label"]["labels"]
    else:
        annotation_types = json.loads(get_finetuned_annotation_types())
        document_labels = annotation_types["document_label"]["labels"]

    number_of_document_labels = len(document_labels)

    # count the number of documents in each document_label class (according to human annotation)
    counter_per_document_label = [0] * number_of_document_labels

    for doc_id in documents:
        human_annotations = get_human_annotations(doc_id)
        for annotation in human_annotations["annotations"]:
            if annotation["annotation_type"] == "document_label":
                counter_per_document_label = [x + y for x, y in zip(counter_per_document_label, annotation["annotation_vector"])] # count up in counter_per_document_label

    #logger.log("number of documents per document label " +str(counter_per_document_label))
    return counter_per_document_label


# counts occurences of human labels per given label_type
def count_per_label(documents, label_type):
    annotation_types = json.loads(get_finetuned_annotation_types())
    labels = annotation_types[label_type]["labels"]

    # count the number of documents in each document_label class (according to human annotation)
    counter_per_label = [0] * len(labels)

    for doc_id in documents:
        human_annotations = get_human_annotations(doc_id)
        for annotation in human_annotations["annotations"]:
            if annotation["annotation_type"] == label_type:
                counter_per_label = [x + y for x, y in zip(counter_per_label, annotation["annotation_vector"])] # count up in counter_per_document_label

    return counter_per_label
    

def exclude_is_other(documents):
    documents_copy = copy.deepcopy(documents)

    for doc_id in documents:
        human_annotations = get_human_annotations(doc_id)  
        for annotation in human_annotations["annotations"]:
            if annotation["annotation_type"] == "document_label": 
                if sum(annotation["annotation_vector"]) == 0: 
                    documents_copy.remove(doc_id)

                    # set target_collection to has_human_annotation_other
                    result = get_document(doc_id)
                    result["target_collection"] = result["target_collection"]+"_other"
                    ds.save_to_collection(result)

    return documents_copy

# exclude documents that do not have a 
# human annotation with annotation_type "document_label"
# i.e. the annotation for document label was not confirmed
# by the user
def exclude_not_confirmed(documents):
    documents_copy = copy.deepcopy(documents)

    for doc_id in documents:
        human_annotations = get_human_annotations(doc_id)  
        has_human_annotation_for_document_label = False
        for annotation in human_annotations["annotations"]:
            if annotation["annotation_type"] == "document_label": 
                has_human_annotation_for_document_label = True

        if has_human_annotation_for_document_label == False: 
            documents_copy.remove(doc_id)

            result = get_document(doc_id)
            result["target_collection"] = result["target_collection"]+"_unconfirmed"
            ds.save_to_collection(result)

    return documents_copy


# take out one batch from <all_except_seen> (only unseen docs, will be added to existing fine-tune batch later)
def get_finetuning_batch(number_of_docs):
    global current_evalutation_scenario
    global test_set
    global initial_finetuning_batch
    global al_finetuning_batches
    global random_finetuning_batches
    global all_except_seen
    global all_except_seen_copy

    fine_tuning_batch = []
    for i in range(number_of_docs):
        next = json.loads(next_most_important("evaluation"))
        if next != None:
            fine_tuning_batch.append(next["document_id"]) 
        
            # remove from <all_except_seen>
            all_except_seen.remove(next["document_id"])
        else:
            logger.log("tried to get next for fine_tuning_batch: received None")

        # do not change target collection because we do fine tuning twice in active learning evaluation (efficient_annotation and random)
        #doc = ds.load_from_datastore_as_json(next["document_id"])
        #doc["target_collection"] = "has_complete_human_annotation_initial_finetune_set"
        #ds.save_to_collection(doc)

    if fine_tuning_batch != []:
        logger.log("generated fine_tuning_batch")
        #logger.log("fine_tuning_batch: " + str(fine_tuning_batch))
        logger.log("all_except_seen: " + str(all_except_seen))
    else:

        logger.log("ERROR: tried to generate fine_tuning_batch but failed")        
        logger.log("all_except_seen: " + str(all_except_seen))

    return fine_tuning_batch
    

@app.route('/start_base_predictions/', methods=['POST'])
def start_base_predictions():
    global current_evalutation_scenario
    global test_set
    global initial_finetuning_batch
    global al_finetuning_batches
    global random_finetuning_batches
    global all_except_seen
    global all_except_seen_copy

    logger.log("start_base_predictions")
    current_evalutation_scenario = "base predictions"

    datasets = [test_set, all_except_seen, initial_finetuning_batch]

    data_exists = False
    for dataset in datasets:
        if dataset != []:
            data_exists = True

    if data_exists == False:
        logger.log("ERROR: no data for prediction or accuracy calculation.")
        return "ERROR: no data for prediction or accuracy calculation."


    for dataset in datasets:
        logger.log("call predict_batch_with_base_model")

        payload = dataset # example: ["id1", "id2", "id3"]
        logger.log("payload: " + str(dataset))

        payload = remove_docs_with_base_prediction(payload)

        if DEBUG_evalutation == False:
            if payload != []:
                #post_json(payload, SEGMENTATION_TOOL_DOCUMENT_PATH +"predict_batch_with_base_model/")
                send_post_requests_with_max_batchsize(payload, SEGMENTATION_TOOL_DOCUMENT_PATH +"predict_batch_with_base_model/")
                logger.log("return from call predict_batch_with_base_model")
            else:
                logger.log("payload is empty. no call to predict_batch_with_base_model")

        if dataset != []:
            calculate_and_store_accuracy(dataset, "base")

    
    logger.log("Done: base_predictions")
    return "Done: base predictions"


def remove_docs_with_base_prediction(payload):
    global WARN_about_has_base

    payload_copy = copy.deepcopy(payload)
    for doc_id in payload:
        result = get_document(doc_id)

        if result != "{}":
            if "has_base" in result and result["has_base"] == True:
                payload_copy.remove(doc_id)
            elif "has_base" not in result and WARN_about_has_base:
                logger.log("ERROR: json does not have key has_base")
                WARN_about_has_base = False
    return payload_copy


@app.route('/start_finetuning_training/', methods=['POST'])
def start_finetuning_training():
    global current_evalutation_scenario
    global test_set
    global initial_finetuning_batch
    global al_finetuning_batches
    global random_finetuning_batches
    global all_except_seen
    global all_except_seen_copy

    logger.log("start_finetuning_training")
    logger.log("call finetune_case_specific_model_on_batch")

    current_evalutation_scenario = "fine-tuning"

    if len(initial_finetuning_batch) <= 1:
        return_message = "ERROR in start_finetuning_training, length of fine_tune_train_batch is <= 1. No call to: " + str(SEGMENTATION_TOOL_DOCUMENT_PATH +"finetune_case_specific_model_on_batch/."
        + "Possible causes: split_dataset was not called before start_finetuning_training. Not enough labeled data (check if there are documents with target_collection = has_complete_human_annotation or has_partial_human_annotation)")
        logger.log(return_message)
        return "ERROR: initial_finetuning_batch is empty."

    payload = initial_finetuning_batch # example: ["id4", "id5"]
    logger.log("fine-tuning batch: " + str(payload))

    if DEBUG_evalutation == False:
        #post_json(payload, SEGMENTATION_TOOL_DOCUMENT_PATH +"finetune_case_specific_model_on_batch/")
        repeated_post_json(payload, SEGMENTATION_TOOL_DOCUMENT_PATH +"finetune_case_specific_model_on_batch/")

    logger.log("return from call finetune_case_specific_model_on_batch")

    # wait for call to train_done
    if DEBUG_evalutation == True:
        train_done()

    after_train_done()

    logger.log("Done: start_finetuning_training")
    return "Done: start_finetuning_training"

  
@app.route('/get_finetuned_annotation_types/', methods=['GET'])
def get_finetuned_annotation_types():
    #logger.log("get_finetuned_annotation_types")
    #logger.log(finetuned_annotation_types)

    return json.dumps(finetuned_annotation_types)


@app.route('/get_filtered_finetuned_annotation_types/', methods=['GET'])
def get_filtered_finetuned_annotation_types():
    #logger.log("get_filtered_finetuned_annotation_types")
    #logger.log("cwd:" + os.getcwd())
    #logger.log("open file at:" + logger.get_path_to_logged_file("filtered_finetuned_annotation_types.json"))

    with open(logger.get_path_to_logged_file("filtered_finetuned_annotation_types.json")) as f:
        d = json.load(f)
        #logger.log(d)
        return json.dumps(d)


@app.route('/reset_target/<path:from_target>/<path:to_target>', methods=['POST'])
def reset_target(from_target, to_target):
    ds.set_target_collection_old_to_new(from_target, to_target)
    return ("Done: set all documents with target_collection =  " + from_target + " to tagret_collection = " + to_target)


# called by AP3, when models are trained on all requested examples
# @app.route('/train_done/', methods=['POST'])
# def train_done():
#     global current_evalutation_scenario
#     global test_set
#     global initial_finetuning_batch
#     global al_finetuning_batches
#     global random_finetuning_batches
#     global all_except_seen
#     global all_except_seen_copy

#     logger.log("train_done")
#     logger.log("current_evalutation_scenario: " + current_evalutation_scenario)

#     # when training is done we can start testing
#     logger.log("get finetuned predictions on test set")
#     start_finetuned_predictions(test_set)
#     calculate_and_store_accuracy(test_set, "finetuned")

#     if current_evalutation_scenario == "fine-tuning":
#         logger.log("Done: evalutation scenario fine-tuning")
#         return "Done: evalutation scenario fine-tuning"

#     return "Done: train done"


@app.route('/train_done', methods=['POST'])
def train_done():
    logger.log("received train_done from AP3")
    return "Done: train done"


def after_train_done():
    logger.log("after receiving train_done: call AP3 to get finetuned predictions on test set")
    start_finetuned_predictions(test_set)
    logger.log("after receiving train_done and receiving finetuned predictions on testset: calculate_and_store_accuracy on test set")
    calculate_and_store_accuracy(test_set, "finetuned")
    logger.log("Done: after train done")
    return "Done: after train done"


# start finetuned predictions on Document_Overview.csv
# skip document ids until given ids
# includes the given id
@app.route('/start_finetuned_predictions/<path:id>', methods=['POST'])
def start_finetuned_predictions_rest_call_with_id(id):

    logger.log("start_finetuned_predictions with DOCUMENT_OVERVIEW_CSV. starting from id " + str(id) + " (included)")

    t_start = time.time()

    AP3_rest_call = "predict_batch_with_latest_finetuned_model/"

    ### read Overview.csv ###
    if DEBUG_prediction == True:
        df = pd.read_csv(r"Document_Overview_sm.csv", sep=DOCUMENT_OVERVIEW_CSV_ENCODING_DELIMITER, encoding=DOCUMENT_OVERVIEW_CSV_ENCODING)
    else:
        df = pd.read_csv(DOCUMENT_OVERVIEW_CSV, sep=DOCUMENT_OVERVIEW_CSV_ENCODING_DELIMITER, encoding=DOCUMENT_OVERVIEW_CSV_ENCODING)

    document_id_list = []
    start_adding_doc_ids = False
    ### iterate over files in Overview.csv and collect all doc_ids ###
    for document_id in df['FileID']:
        if document_id == id:
            start_adding_doc_ids = True

        if start_adding_doc_ids:
            document_id_list.append(document_id)

    logger.log("call " + AP3_rest_call)

    send_post_requests_with_max_batchsize(document_id_list, SEGMENTATION_TOOL_DOCUMENT_PATH + AP3_rest_call) 

    logger.log("return from call " + AP3_rest_call)

    if len(document_id_list) <= 0:
        logger.log("Id "+str(id)+" was not found in DOCUMENT_OVERVIEW_CSV. Parameter must be id in DOCUMENT_OVERVIEW_CSV.")
        message = "ERROR: DOCUMENT_OVERVIEW_CSV has length 0. Did not call " + AP3_rest_call
        logger.log(message)
        return message
    else:
        logger.log(document_id_list)

    t_end = time.time()

    message = ("start_finetuned_predictions done. Sent " + str(len(document_id_list)) 
    + " ids to "+ SEGMENTATION_TOOL_DOCUMENT_PATH + AP3_rest_call
    + "; total time elapsed (for sending all requests): " + str(t_end-t_start) 
    + "; avg. time per doc: " + str((t_end-t_start) / len(document_id_list)))

    logger.log(message)

    return message


# if payload exist: call prediction for all document_ids in payload
# if no payload: call prediction on all documents in Document_Overview.csv
# payload should look like: ['21318f6e12c83acd98a0d1ab33a64617', 'c8488ffbe30a0ba7c6cdffdeb95bb475', '1f766caa049cf705174b1f1aa5baa85e']
@app.route('/start_finetuned_predictions/', methods=['POST'])
def start_finetuned_predictions_rest_call():
    logger.log("start_finetuned_predictions")

    t_start = time.time()

    AP3_rest_call = "predict_batch_with_latest_finetuned_model/"
    
    payload = request.get_json() 

    document_id_list = []

    if payload != {}:
        if len(payload) > 0:
            document_id_list = payload
            logger.log("start_finetuned_predictions on payload: " + str(payload))
            logger.log("call " + AP3_rest_call)

            #if DEBUG_prediction == False:
            #    post_json(payload, SEGMENTATION_TOOL_DOCUMENT_PATH + AP3_rest_call)
            send_post_requests_with_max_batchsize(document_id_list, SEGMENTATION_TOOL_DOCUMENT_PATH + AP3_rest_call) 
            logger.log("return from call "+AP3_rest_call)
        else:
            message = "ERROR: payload for start_finetuned_predictions has length 0. Did not call " + AP3_rest_call
            logger.log(message)
            return message
    else:

        ### read Overview.csv ###
        if DEBUG_prediction == True:
            df = pd.read_csv(r"Document_Overview_sm.csv", sep=DOCUMENT_OVERVIEW_CSV_ENCODING_DELIMITER, encoding=DOCUMENT_OVERVIEW_CSV_ENCODING)
        else:
            df = pd.read_csv(DOCUMENT_OVERVIEW_CSV, sep=DOCUMENT_OVERVIEW_CSV_ENCODING_DELIMITER, encoding=DOCUMENT_OVERVIEW_CSV_ENCODING)
    
        ### iterate over files in Overview.csv and collect all doc_ids###
        for document_id in df['FileID']:
            document_id_list.append(document_id)

        logger.log("start_finetuned_predictions on DOCUMENT_OVERVIEW_CSV. document_ids: " + str(document_id_list))

        ### send request to AP4 to start prediction on document with document_id ###
        logger.log("call " + AP3_rest_call)
        #if DEBUG_prediction == False:
        #    if len(document_id_list) > 0:
        #        post_json(document_id_list, SEGMENTATION_TOOL_DOCUMENT_PATH + AP3_rest_call)
        
        send_post_requests_with_max_batchsize(document_id_list, SEGMENTATION_TOOL_DOCUMENT_PATH + AP3_rest_call) 
        logger.log("return from call " + AP3_rest_call)

        if len(document_id_list) <= 0:
            message = "ERROR: DOCUMENT_OVERVIEW_CSV has length 0. Did not call " + AP3_rest_call
            logger.log(message)
            return message

    t_end = time.time()

    message = ("start_finetuned_predictions done. Sent " + str(len(document_id_list)) 
    + " ids to "+ SEGMENTATION_TOOL_DOCUMENT_PATH + AP3_rest_call
    + "; total time elapsed (for sending all requests): " + str(t_end-t_start) 
    + "; avg. time per doc: " + str((t_end-t_start) / len(document_id_list)))

    logger.log(message)
    return message


# TODO rename method for clarity
def start_finetuned_predictions(dataset):
    global current_evalutation_scenario
    global test_set
    global initial_finetuning_batch
    global al_finetuning_batches
    global random_finetuning_batches
    global all_except_seen
    global all_except_seen_copy

    # call with test set for testing 
    logger.log("start_finetuned_predictions")
    logger.log("current_evalutation_scenario: " + current_evalutation_scenario)

    if current_evalutation_scenario == "fine-tuning":
        send_post_requests_with_max_batchsize(dataset, SEGMENTATION_TOOL_DOCUMENT_PATH +"predict_batch_with_base_finetuned_model/")

    elif current_evalutation_scenario == "active learning efficient_annotation":
       send_post_requests_with_max_batchsize(dataset, SEGMENTATION_TOOL_DOCUMENT_PATH +"predict_batch_with_latest_finetuned_model/")

    elif current_evalutation_scenario == "active learning efficient_annotation use base finetuned":
      send_post_requests_with_max_batchsize(dataset, SEGMENTATION_TOOL_DOCUMENT_PATH +"predict_batch_with_base_finetuned_model/")

    logger.log("Done: finetuned predictions")
    return "Done: finetuned predictions"




# call 1x with type efficient_annotation and 1x random
@app.route('/start_active_learning/<path:type>', methods=['POST'])
def start_active_learning(type):
    global current_evalutation_scenario
    global test_set
    global initial_finetuning_batch
    global al_finetuning_batches
    global random_finetuning_batches
    global all_except_seen
    global all_except_seen_copy


    if len(initial_finetuning_batch) == 0:
        logger.log("ERROR: start_active_learning: initial_finetuning_batch is empty.")
        return "ERROR: initial_finetuning_batch is empty."

    # remove one trailing /
    if len(type) > 0 and type[-1] == "/":
        type = type[0:-1]

    logger.log("start_active_learning with type "+ str(type))

    print("start_active_learning with type "+ str(type))
    if type != "efficient_annotation" and type != "random":
        logger.log("ERROR: Wrong type. Please start active learning evalutation using "+
        "start_active_learning/efficient_annotation or start_active_learning/random")

        return("ERROR: Wrong type. Please start active learning evalutation using "+
        "start_active_learning/efficient_annotation or start_active_learning/random")

    else:
        current_evalutation_scenario = "active learning " + type

    if type == "random":
        logger.log("init_next_most_important eval_random")
        init_next_most_important("eval_random") 

    elif type == "efficient_annotation": 
        logger.log("init_next_most_important eval_al")
        init_next_most_important("eval_al")

    if all_except_seen == []: 
        all_except_seen = all_except_seen_copy

    # 1. get predictions for all unseen documents (not necessary for random)
    if type == "efficient_annotation":
        # for first iteration use base fine-tuned model
        current_evalutation_scenario = current_evalutation_scenario + " use base finetuned"
        start_finetuned_predictions(all_except_seen)
        current_evalutation_scenario = "active learning " + type

    number_of_documents_left_for_finetuning = len(all_except_seen)

    # TODO change ratios?
    # 2. select a training batch with active learning
    ratios = config["evaluation"]["active_learning_ratios"]

    if sum(ratios) > 1:
        return_message = "ERROR: sum of active_learning_ratios must be <= 1. Set in config.js evalutation > active_learning_ratios."
        logger.log(return_message)
        return return_message

    logger.log("will split all_except_unseen with the following ratios: "+ str(ratios))
    for idx, ratio in enumerate(ratios):
        number_of_docs = floor(number_of_documents_left_for_finetuning * ratio)
        logger.log("ratio = "+ str(ratio) + " => "+ str(number_of_docs) + " out of " + str(number_of_documents_left_for_finetuning) + " documents.")

    for idx, ratio in enumerate(ratios): # previous batches are always added so final batch is 100% of training data
        logger.log("current_evalutation_scenario: "+ current_evalutation_scenario)

        # put <finetune_split> percent of documents in fine_tune_train_batches
        # and split further into batches (select with active learning)
        
        # put <all_except_seen> into AL queue 
        # take out one batch, re-training, take another batch, retrain etc.
        # take 14% 43% 100% of queue #
        # those are now the finetuning batches 
        # the rest should be predicted (all_except_seen)

        # take finetuning batch from queue
        update_scores_in_current_queue()
        
        number_of_docs = floor(number_of_documents_left_for_finetuning * ratio)
        logger.log("ratio: "+ str(ratio))
        logger.log("add: "+ str(number_of_docs) +" new documents to finetuning batch out of "+ str(number_of_documents_left_for_finetuning) + " documents in all_except_seen")

        finetuning_batch = get_finetuning_batch(number_of_docs)
        
        #  include previous batches in new batch
        logger.log("newly added to finetune_batch: "+ str(finetuning_batch))

        if type == "efficient_annotation":
            fine_tune_train_batches = al_finetuning_batches
        else:            
            fine_tune_train_batches = random_finetuning_batches

        if DEBUG_evalutation:
            # make sure the newly added documents are not repeated
            for batch in fine_tune_train_batches:
                for doc_id in batch:
                    for added_doc_id in finetuning_batch:
                        if doc_id == added_doc_id:
                            print("ERROR: doc_id "+str(added_doc_id)+" already in fine_tune_train_batches")

        if idx == 0:
            previous_finetune_batch = initial_finetuning_batch
        else:
            previous_finetune_batch = fine_tune_train_batches[-1]


        # extend finetune_batch with previous batches
        finetuning_batch.extend(previous_finetune_batch)
        fine_tune_train_batches.append(finetuning_batch) 

        # retrain with finetuning batch
        payload = finetuning_batch # example: ["id4", "id5","id6"]

        if type == "efficient_annotation":
            al_finetuning_batches = fine_tune_train_batches
        else:            
            random_finetuning_batches = fine_tune_train_batches

        logger.log("call train_case_specific_model_on_batch")
        logger.log("finetuning_batch: "+ str(payload))
        logger.log("finetuning_batch len: " + str(len(payload)))
        logger.log(str(payload), "finetuning_batch_" + str(current_evalutation_scenario) +str(len(fine_tune_train_batches))+".txt","w")
        logger.log(str(len(payload)), "finetuning_batch_" + str(current_evalutation_scenario) +str(len(fine_tune_train_batches))+".txt","a")

        if DEBUG_evalutation == False:
            if len(payload) > 1:
                #post_json(payload, SEGMENTATION_TOOL_DOCUMENT_PATH +"train_case_specific_model_on_batch/")
                repeated_post_json(payload, SEGMENTATION_TOOL_DOCUMENT_PATH +"train_case_specific_model_on_batch/")
            else:
                logger.log("ERROR: did not send request to AP4 train_case_specific_model_on_batch. Length of payload must be > 1.")
        logger.log("return from call train_case_specific_model_on_batch")

         
        # 3. wait for call to train_done to test ml models #
        if DEBUG_evalutation == True:
            train_done()

        after_train_done()

        # make sure that all_except_seen is empty when one round of active learning is finished
        # so that we fill it with copy when we do another round of active learning
        if idx == (len(ratios)-1):
            all_except_seen = []

    logger.log("Done: active learning " + str(type))
    return "Done: active learning"


# when prediction is finished on test set
# compare human annotations to predictions
def calculate_and_store_accuracy(dataset, finetuned_or_base_model):
    logger.log("calculate_and_store_accuracy for document_label")
    logger.log(("current_evalutation_scenario: " + str(current_evalutation_scenario)), "accuracy.txt", "a")
    logger.log(("current_evalutation_scenario: " + str(current_evalutation_scenario)))

    logger.log(("finetuned_or_base_model? " + str(finetuned_or_base_model)), "accuracy.txt", "a")

    accuracies = dict()

    # ignore segment boundary annotations
    # always use most recent annotation_version of ml_annotation and human_annotation

    if current_evalutation_scenario == "base predictions":
        annotation_types = json.loads(get_annotation_types())
        document_labels = annotation_types["document_label"]["labels"]
        len_document_labels = len(document_labels)

    accuracy = 0
    n = 0
    for doc_id in dataset:
        ml_annotations = get_ml_annotations(doc_id)
        for ml_annotation in ml_annotations["annotations"]:
            if (ml_annotation["annotation_type"]  != "segment_boundary" and not (finetuned_or_base_model == "finetuned" and ml_annotation["annotator_type"] == "model_base")):

                human_annotations = get_human_annotations(doc_id)
                for human_annotation in human_annotations["annotations"]:

                    # compare doc_label
                    if human_annotation["annotation_type"] == "document_label" and ml_annotation["annotation_type"]  == "document_label": 
                        logger.log("document_id: " + str(doc_id), "accuracy.txt","a")

                        logger.log(human_annotation["annotation_type"], "accuracy.txt","a")
                        logger.log("ml_annotation annotator_id: " + str(ml_annotation["annotator_id"]), "accuracy.txt","a")
                        logger.log("ml_annotation annotator_type:" + str(ml_annotation["annotator_type"]), "accuracy.txt","a")

                        logger.log("human_annotation      :" + str(human_annotation["annotation_vector"]), "accuracy.txt","a")
                        logger.log("ml_annotation         :" + str(ml_annotation["annotation_vector"]), "accuracy.txt","a")
                        

                        if current_evalutation_scenario == "base predictions":
                            # in this evalutation_scenario the ml annotation_vectors for document_label are 
                            # in the annotation_types format i.e. they have length 19
                            # while the human annotation are in the finetuned_annotation_types format (length 28) 
                            # the first 19 labels are the same and can be compared - the others will be ignored

                            if len(ml_annotation["annotation_vector"]) != len_document_labels:
                                logger.log("#### WARNING: ml annotation_vector document_label of base model does not have length as specified in annotation types ####")

                            human_annotation_short = human_annotation["annotation_vector"]
                            human_annotation_short = human_annotation_short[0:len_document_labels]
                            #logger.log("human_annotation_short:" + str(human_annotation_short) , "accuracy.txt","a")
                            
                            # use shortened vector to compute accuracy
                            logger.log("accurate: " + str(get_accuracy_of_two_vectors(human_annotation_short, ml_annotation["annotation_vector"]))  + "\n", "accuracy.txt","a")
                            accuracy = get_accuracy_of_two_vectors(human_annotation_short, ml_annotation["annotation_vector"])
                            accuracies = add_to_accuracies(accuracies, ml_annotation, accuracy)

                        else:
                            logger.log("accurate: " + str(get_accuracy_of_two_vectors(human_annotation["annotation_vector"], ml_annotation["annotation_vector"]))+ "\n", "accuracy.txt","a")
                            accuracy = get_accuracy_of_two_vectors(human_annotation["annotation_vector"], ml_annotation["annotation_vector"])
                            accuracies = add_to_accuracies(accuracies, ml_annotation, accuracy)


                    # compare doc_type
                    elif human_annotation["annotation_type"] == "document_type" and ml_annotation["annotation_type"]  == "document_type": 
                        logger.log("document_id: " + str(doc_id), "accuracy.txt","a")
                        logger.log(human_annotation["annotation_type"], "accuracy.txt","a")
                        logger.log("ml_annotation annotator_id: " + str(ml_annotation["annotator_id"]), "accuracy.txt","a")
                        logger.log("ml_annotation annotator_type:" + str(ml_annotation["annotator_type"]), "accuracy.txt","a")

                        logger.log("human_annotation      :" + str(human_annotation["annotation_vector"]), "accuracy.txt","a")
                        logger.log("ml_annotation         :" + str(ml_annotation["annotation_vector"]), "accuracy.txt","a")

                        accuracy = get_accuracy_of_two_vectors(human_annotation["annotation_vector"], ml_annotation["annotation_vector"])
                        accuracies = add_to_accuracies(accuracies, ml_annotation, accuracy)
                        
                        logger.log("accurate: " + str(get_accuracy_of_two_vectors(human_annotation["annotation_vector"], ml_annotation["annotation_vector"]))+ "\n", "accuracy.txt","a")


                    # compare segments
                    elif human_annotation["segment_id"] != "" and (human_annotation["segment_id"] == ml_annotation["segment_id"]):
                        logger.log("document_id: " + str(doc_id), "accuracy.txt","a")
                        logger.log(human_annotation["annotation_type"], "accuracy.txt","a")
                        logger.log("ml_annotation annotator_id: " + str(ml_annotation["annotator_id"]), "accuracy.txt","a")
                        logger.log("ml_annotation annotator_type:" + str(ml_annotation["annotator_type"]), "accuracy.txt","a")

                        logger.log("human_annotation      :" + str(human_annotation["annotation_vector"]), "accuracy.txt","a")
                        logger.log("ml_annotation         :" + str(ml_annotation["annotation_vector"]), "accuracy.txt","a")
                        
                        logger.log("accurate: " + str(get_accuracy_of_two_vectors(human_annotation["annotation_vector"], ml_annotation["annotation_vector"]))+ "\n", "accuracy.txt","a")

                        accuracy = get_accuracy_of_two_vectors(human_annotation["annotation_vector"], ml_annotation["annotation_vector"])
                        accuracies = add_to_accuracies(accuracies, ml_annotation, accuracy)

            #else:
                #logger.log("ml_annotation annotation_type is segment boundary -> ignore")
                #logger.log("document_id: " + str(doc_id), "accuracy.txt","a")
                #logger.log(ml_annotation["annotation_type"], "accuracy.txt","a")
                #logger.log("ml_annotation annotator_id: " + str(ml_annotation["annotator_id"]), "accuracy.txt","a")
                #logger.log("ml_annotation annotator_type:" + str(ml_annotation["annotator_type"]), "accuracy.txt","a")
                #logger.log("ml_annotation         :" + str(ml_annotation["annotation_vector"]), "accuracy.txt","a")
                

    for entry in accuracies:
        logger.log("type: " + str(entry), "accuracy.txt","a")
        logger.log("type: " + str(entry))

        sum = accuracies[entry]["sum"]
        n = accuracies[entry]["n"]

        if n > 0:
            avg_accuracy = (sum / n)

        logger.log("average accuracy with " + str(n) + " samples: "  + str(avg_accuracy), "accuracy.txt","a")
        logger.log("average accuracy with " + str(n) + " samples: " + str(avg_accuracy))
            
    logger.log("dataset: " + str(dataset) + "\n", "accuracy.txt","a")
    logger.log("dataset: " + str(dataset) + "\n")


def add_to_accuracies(accuracies, ml_annotation, accuracy):
    type = ml_annotation["annotation_type"]+"_"+ml_annotation["annotator_id"]

    if type not in accuracies: 
        accuracies[type] = {"sum": accuracy, "n":1}
    else:
        accuracies[type] = {"sum": accuracies[type]["sum"] + accuracy, "n":accuracies[type]["n"] + 1}
        logger.log(accuracies, "accuracy.txt","a")

    return accuracies


# compares if the two vectors have their maximum value at the same position
# if the maximum value occurs twice in the same vector, accuracy will be 0
def get_accuracy_of_two_vectors(vec1, vec2):

    if len(vec1) != len(vec2):
        logger.log(current_evalutation_scenario)
        logger.log("human: " + str(vec1))
        logger.log("ml: " + str(vec2))

        logger.log("ERROR: vectors do not have same length - cannot compute accuracy.")
        logger.log("ERROR: vectors do not have same length - cannot compute accuracy.", "accuracy.txt","a")
        logger.log("human: " + str(vec1) + "\n" + "len = " + str(len(vec1)), "accuracy.txt","a")
        logger.log("ml: " + str(vec2)+ "\n" + "len = " + str(len(vec2)), "accuracy.txt","a")

        return 0

    m1 = max(vec1)
    index1 = [i for i, j in enumerate(vec1) if j == m1]

    m2 = max(vec2)
    index2 = [i for i, j in enumerate(vec2) if j == m2]

    #print(vec1, flush = True)
    #print(vec2, flush = True)
    #print(index1, flush = True)
    #print(index2, flush = True)

    if len(index1) > 1 or len(index2) > 1:
        return 0

    if index1[0] == index2[0]: 
        return 1 # both vectors agree on the same class
    else:
        return 0


# Example output:
# { annotations:
# [
# {'annotation_id': 'KLVCM1Ul', 'annotator_id': 'annotator1', 'annotator_type': 'human', 'annotation_vector': [0, 0, 1, 0], 'annotation_type': 'segment_type', 'segment_id': '821lqu3a', 'document_id': '7bed28f273d108e742e351d191b05bc0'},
# {'annotation_id': 'gvzUDfzi', 'annotator_id': 'annotator1', 'annotator_type': 'human', 'annotation_vector': [0, 1], 'annotation_type': 'is_occluded', 'segment_id': 'k5J0S6Tt', 'document_id': '7bed28f273d108e742e351d191b05bc0'},
# ...
# ]
# } 
@app.route('/get_human_annotations/<path:document_id>', methods=['GET'])
def get_human_annotations(document_id):

    document_json = get_document(document_id)
    if document_json == "{}":
        logger.log("ERROR in get_human_annotations document_id " + str(document_id) + " not found.")
        return "404 Not Found"
        

    output_json = {}
    output_json["annotations"] = []

    # convert to Document object
    document_obj = Document.deserialize(document_json) 

    # add document label annotation
    for annotation in document_obj.document_label.annotations:

        if not hasattr(annotation, "annotation_version"):
            global WARN_about_annotation_version
            if WARN_about_annotation_version:
                logger.log("ERROR: no annotation_version in annotation e.g. in document " + str(document_id) + " set annotation_version to -1")
                WARN_about_annotation_version = False
            setattr(annotation, "annotation_version", -1)


        if annotation.annotator_type == "human":
            a = {
            "annotation_id" :  annotation.annotation_id, 
            "annotation_version" :  annotation.annotation_version,   
            "annotator_id" :  annotation.annotator_id,
            "annotator_type" : annotation.annotator_type,
            "annotation_vector" : annotation.annotation_vector,
            "annotation_type" : annotation.annotation_type,
            "segment_id" : "",
            "document_id" : document_id,
            "page_number" : "",
            }
            #print(a, flush = True)
            output_json["annotations"].append(a)

    # add document type annotation
    for annotation in document_obj.document_type.annotations:
        if not hasattr(annotation, "annotation_version"):
            setattr(annotation, "annotation_version", -1)

        if annotation.annotator_type == "human":
            a = {
            "annotation_id" :  annotation.annotation_id, 
            "annotation_version" :  annotation.annotation_version,     
            "annotator_id" :  annotation.annotator_id,
            "annotator_type" : annotation.annotator_type,
            "annotation_vector" : annotation.annotation_vector,
            "annotation_type" : annotation.annotation_type,
            "segment_id" : "",
            "document_id" : document_id,
            "page_number" : "",
            }
            #print(a, flush = True)
            output_json["annotations"].append(a)

    # add segment annotations
    for page in document_obj.pages:
        for segment in page.segments:
            for annotation in segment.annotations():
                if not hasattr(annotation, "annotation_version"):
                    #logger.log("ERROR: no annotation_version in annotation in document " + str(document_id))
                    setattr(annotation, "annotation_version", -1)

                if annotation.annotator_type == "human":
                    a = {
                    "annotation_id" :  annotation.annotation_id,  
                    "annotation_version" :  annotation.annotation_version,    
                    "annotator_id" :  annotation.annotator_id,
                    "annotator_type" : annotation.annotator_type,
                    "annotation_vector" : annotation.annotation_vector,
                    "annotation_type" : annotation.annotation_type,
                    "segment_id" : segment.segment_id,
                    "document_id" : document_id,
                    "page_number" : page.page_number,
                    }
                    #print(a, flush = True)
                    output_json["annotations"].append(a) 


    # in output_json remove annotations of the same segment or document_label, document_type
    # that have the same annotator_id but have a lower version
    for annotation1 in output_json["annotations"]:
        for annotation2 in output_json["annotations"]:

            if (annotation1["annotation_type"]  == "document_type" and annotation2["annotation_type"]  == "document_type" and annotation1["annotation_version"]  >  annotation2["annotation_version"] ):
                annotation2["annotation_id"] = "remove"
                #output_json_copy["annotations"].remove(annotation2)


            elif (annotation1["annotation_type"] == "document_label"  and annotation2["annotation_type"] == "document_label" and annotation1["annotation_version"]  >  annotation2["annotation_version"]):
                annotation2["annotation_id"] = "remove"
                #output_json["annotations"].remove(annotation2)
                
            # for segments
            elif (annotation1["segment_id"] == annotation2["segment_id"] and annotation1["annotation_version"]  >  annotation2["annotation_version"]):
                annotation2["annotation_id"] = "remove"
                #output_json["annotations"].remove(annotation2)

    output_json_copy = copy.deepcopy(output_json)

    for annotation in output_json["annotations"]:
        if annotation["annotation_id"] == "remove":
            print("remove", annotation["annotation_id"], flush = True)
            output_json_copy["annotations"].remove(annotation)

    return output_json_copy                    


@app.route('/get_ml_annotations/<path:document_id>', methods=['GET'])
def get_ml_annotations(document_id):
    output_json = {}
    output_json["annotations"] = []

    document_json = get_document(document_id)
    if document_json == "{}":
        logger.log("WARNING: document with document_id " + str(document_id) + "not found. get_ml_annotations returns " + str(output_json))
        return output_json

    # convert to Document object
    document_obj = Document.deserialize(document_json) 

    ml_annotator_ids = ["segment_multimodal_model", "segment_visual_model", "segment_text_model", "segment_detection_model"]

    # add document label annotation
    for annotation in document_obj.document_label.annotations:
        if not hasattr(annotation, "annotation_version"):
            setattr(annotation, "annotation_version", -1)

        if annotation.annotator_type != "human":
            a = {
            "annotation_id" :  annotation.annotation_id, 
            "annotation_version" :  annotation.annotation_version,     
            "annotator_id" :  annotation.annotator_id,
            "annotator_type" : annotation.annotator_type,
            "annotation_vector" : annotation.annotation_vector,
            "annotation_type" : annotation.annotation_type,
            "segment_id" : "",
            "document_id" : document_id,
            "page_number" : ""
            }
            #print(a, flush = True)
            output_json["annotations"].append(a)

    # add document type annotation
    for annotation in document_obj.document_type.annotations:
        if not hasattr(annotation, "annotation_version"):
            setattr(annotation, "annotation_version", -1)

        if annotation.annotator_type != "human":
            a = {
            "annotation_id" :  annotation.annotation_id,  
            "annotation_version" :  annotation.annotation_version,    
            "annotator_id" :  annotation.annotator_id,
            "annotator_type" : annotation.annotator_type,
            "annotation_vector" : annotation.annotation_vector,
            "annotation_type" : annotation.annotation_type,
            "segment_id" : "",
            "document_id" : document_id,
            "page_number" : ""
            }
            #print(a, flush = True)
            output_json["annotations"].append(a)

    # add segment annotations
    for page in document_obj.pages:
        for segment in page.segments:
            for annotation in segment.annotations():
                if not hasattr(annotation, "annotation_version"):
                    setattr(annotation, "annotation_version", -1)

                if annotation.annotator_type != "human":
                    a = {
                    "annotation_id" :  annotation.annotation_id,   
                    "annotation_version" :  annotation.annotation_version,   
                    "annotator_id" :  annotation.annotator_id,
                    "annotator_type" : annotation.annotator_type,
                    "annotation_vector" : annotation.annotation_vector,
                    "annotation_type" : annotation.annotation_type,
                    "segment_id" : segment.segment_id,
                    "document_id" : document_id,
                    "page_number" : page.page_number
                    }
                    #print(a, flush = True)
                    output_json["annotations"].append(a) 



    # in first pass: set annotation_id  of annotations that should be removed to remove
    # in second pass: remove those annotations

    # in output_json remove annotations of the same segment or document_label, document_type
    # that have the same annotator_id but have a lower version
    for annotation1 in output_json["annotations"]:
        for annotation2 in output_json["annotations"]:

            if annotation1["annotator_id"] == annotation2["annotator_id"]:

                if (annotation1["annotation_type"]  == "document_type" and annotation2["annotation_type"]  == "document_type" and annotation1["annotation_version"]  >  annotation2["annotation_version"] ):
                    annotation2["annotation_id"] = "remove"
                    #output_json_copy["annotations"].remove(annotation2)


                elif (annotation1["annotation_type"] == "document_label"  and annotation2["annotation_type"] == "document_label" and annotation1["annotation_version"]  >  annotation2["annotation_version"]):
                    annotation2["annotation_id"] = "remove"
                    #output_json["annotations"].remove(annotation2)
                  
                # for segments
                elif (annotation1["segment_id"] == annotation2["segment_id"] and annotation1["annotation_version"]  >  annotation2["annotation_version"]):
                    annotation2["annotation_id"] = "remove"
                    #output_json["annotations"].remove(annotation2)

    output_json_copy = copy.deepcopy(output_json)

    for annotation in output_json["annotations"]:
        if annotation["annotation_id"] == "remove":
            print("remove", annotation["annotation_id"], flush = True)
            output_json_copy["annotations"].remove(annotation)


    return output_json_copy


@app.route('/store_db_to_folders/', methods=['POST'])
def store_db_to_folders():
    logger.log("start: store_db_to_folders")
    clean_collection_folders()

    ds.save_all_in_db_to_folders()
    logger.log("Done: store_db_to_folders")

    return "Done storing database to folders."

# remove document-files that are not in the correct collection folders
@app.route('/clean_collection_folders/', methods=['POST'])
def clean_collection_folders():
    ds.clean_collection_folders()
    return "Done cleaning collection folders."



# return most recent finetune batch
#@app.route('/get_mm_split_ids/', methods=['GET'])
#def get_mm_split_ids():
#    if len(fine_tune_train_batches) > 0:
#        return fine_tune_train_batches[-1]
#    else:
#        logger.log("ERROR: in get_mm_split_ids: fine_tune_train_batches empty.")
#        return []

@app.route('/get_dataset_split/', methods=['GET'])
def get_dataset_split():

    global current_evalutation_scenario
    global test_set
    global initial_finetuning_batch
    global al_finetuning_batches
    global random_finetuning_batches
    global all_except_seen
    global all_except_seen_copy

    output_json = {}
    output_json["test_set"] = {"ids" : test_set, "length" : len(test_set)}

    f_lengths = []
    for batch in al_finetuning_batches:
        f_lengths.append(len(batch))
    output_json["al_finetuning_batches"] = {"ids" : al_finetuning_batches, "length" : f_lengths}
    
    f_lengths = []
    for batch in random_finetuning_batches:
        f_lengths.append(len(batch))
    output_json["random_finetuning_batches"] = {"ids" : random_finetuning_batches, "length" : f_lengths}

    output_json["initial_finetuning_batch"] = {"ids" : initial_finetuning_batch, "length" : len(initial_finetuning_batch)}
    output_json["all_except_seen"] = {"ids" : all_except_seen, "length" : len(all_except_seen)}

    return output_json


# {partial: [], complete: [], total: []}
@app.route('/get_document_label_distribution/', methods=['GET'])
def get_document_label_distribution():
    label_type = "document_label"    
    return json.dumps(get_label_distribution(label_type))
    
# {partial: [], complete: [], total: []}
@app.route('/get_segment_label_distribution/', methods=['GET'])
def get_segment_label_distribution():
    label_type = "segment_label"    
    return json.dumps(get_label_distribution(label_type))

# {partial: [], complete: [], total: []}
@app.route('/get_segment_type_distribution/', methods=['GET'])
def get_segment_type_distribution():
    label_type = "segment_type"    
    return json.dumps(get_label_distribution(label_type))


# {partial: [], complete: [], total: []}
@app.route('/get_document_type_distribution/', methods=['GET'])
def get_document_type_distribution():
    label_type = "document_type"    
    return json.dumps(get_label_distribution(label_type))


@app.route('/get_label_and_type_distribution/', methods=['GET'])
def get_label_and_type_distribution():

    label_distribution = {}
    for label_type in ["segment_type", "document_type", "segment_label", "document_label"]:    
        label_distribution[label_type] = get_label_distribution(label_type)
    
    return json.dumps(label_distribution)



# {partial: [], complete: [], total: []}
def get_label_distribution(label_type):
    label_distribution = {}
    label_distribution[label_type+" partial"] = []
    label_distribution[label_type+" complete"] = []
    label_distribution[label_type+" total"] = []

    all_documents_with_partial_human_annotation = []
    all_documents_with_complete_human_annotation = []

    collection_list_partial = []
    collection_list_complete = []


    for tag in ["","_test_set", "_initial_finetune_set", "_other", "_not_enough_examples", "_unconfirmed"]:
        collection_list_partial.append({"target_collection":("has_partial_human_annotation"+tag)})
        collection_list_complete.append({"target_collection":("has_complete_human_annotation"+tag)})
    
    # ---- partial documents ----

    for doc in ds.document_generator_from_query({"$or":  collection_list_partial}, False):
        all_documents_with_partial_human_annotation.append(doc.document_id)

    label_distribution[label_type+" partial"] = count_per_label(all_documents_with_partial_human_annotation, label_type)


    # ---- complete documents ----
    for doc in ds.document_generator_from_query({"$or":  collection_list_complete}, False):
        all_documents_with_complete_human_annotation.append(doc.document_id)

    label_distribution[label_type+" complete"] = count_per_label(all_documents_with_complete_human_annotation, label_type)

    # total
    label_distribution[label_type+" total"] = [sum(x) for x in zip(label_distribution[label_type+" partial"], label_distribution[label_type+" complete"])] 
    

    logger.log(label_distribution)
    return label_distribution




# update list only if 10 secs passed
other_labels_list = {}
timer_update_other_labels = -1

# example:
#{"segment_label":["Text", "Zahlungsbedingungen", ...], 
# "document_label":["Email", "Ueberweisung" ...], 
# "segment_type":["is_other"],
# "document_type": ["is_other"]}
@app.route('/get_other_labels/', methods=['GET'])
def get_other_labels():
    global other_labels_list
    global timer_update_other_labels


    if timer_update_other_labels == -1 or time.time() > (timer_update_other_labels + 10):
        timer_update_other_labels = time.time()

        other_labels_list["document_label"] = []
        other_labels_list["document_type"] = []
        other_labels_list["segment_label"] = []
        other_labels_list["segment_type"] = []

        for other_label in json.loads(count_other_labels()):
            other_labels_list[other_label["type"]].append(other_label["name"])

    return other_labels_list


# example:
#[{"type": "segment_label", "num": 15, "name": "Text"}, 
# {"type": "segment_label", "num": 12, "name": "Tabelle"}, 
# {"type": "segment_label", "num": 123, "name": "is_other"}, 
# {"type": "document_label", "num": 30, "name": "is_other"}, ...]
@app.route('/count_other_labels/', methods=['GET'])
def count_other_labels():

    # other-labels for document_labels, document_type, segment_label, segment_type
    is_other_list = []

    # get documents that are not in ml_out 
   
    for document_json in get_documents_by_query_as_cursor({"target_collection":{"$ne": "ml_out"}}): 

        # convert to Document object
        document_obj = Document.deserialize(document_json) 

        types = []
        types.extend(document_obj.document_label.annotations)
        types.extend(document_obj.document_type.annotations)
        
        # add document label and document type
        for page in document_obj.pages:
            for segment in page.segments:
                types.extend(segment.annotations())

        for annotation in types:
            # only use annotations written by humans, where is_other field exists and is not empty 
            if annotation.annotator_type == "human" and ("is_other" in annotation.__dict__) and annotation.is_other != "":
                name = annotation.is_other
                type = annotation.annotation_type
                already_exists = False
                for item in is_other_list:
                    if item["name"] == name and item["type"] == type:
                        item["num"] = item["num"]+1
                        already_exists = True
                        break

                if already_exists == False:
                   is_other_list.append({"type": type, "num":1, "name":name})


    return json.dumps(is_other_list)


@app.route('/get_blocked_ids/', methods=['GET'])
def get_blocked_ids():
    return json.dumps(ds.get_blocked_ids())





#### end: REST calls used for evaluation    





#app.run(host="0.0.0.0", port="5000")


