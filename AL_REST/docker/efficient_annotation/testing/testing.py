import json
import os.path
import pandas as pd
import urllib
import time
import shutil 
import traceback
import random
import urllib
import threading

from efficient_annotation.common import post_json, get_json, Document
from efficient_annotation.datastores import JSONDatastore
from efficient_annotation.datastores import MongoDB
from efficient_annotation.queuing import QueueManager
from efficient_annotation.common import load_config_file
from efficient_annotation import reformat_jsons

# testing module
print('start testing')
print(os.getcwd())
config = json.load(open("config.json", "r", encoding="utf-8"))


NO_CLEANUP_AFTER = False

# <data_path> points to an empty folder 
# during execution of testing.py,
# files in <test_data> are temporarily copied to <data_path>

data_path = "C:\\Users\\anna_\OneDrive\\Documents\\ActiveLearning\\AL_REST\\docker\\mongodb\\"
#test_data = "..\\20200610_krex-activelearning-resultonly"
test_data = "..\\test_data" # contains ml_out data
base_url = "http://localhost:5000"

annotator_ids = ["annotator_0", "annotator_1", "annotator_2"]

"""
def create_test_environment():
    data_dirs = os.listdir(data_path)
    names = []
    for d in data_dirs:
        d = os.path.join(data_path, d)
        os.rename(d, d+"_tmp")
        os.mkdir(d)
        names.append((d, d+"_tmp"))
    return names
"""

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def cleanup(timepoint):

    print("cleanup: empty db")
    post_json({}, base_url+"/empty_db/")

    if timepoint == "start":
        print("cleanup: empty log")
        empty_log()

    data_dirs = os.listdir(data_path)
    #print('remove all files from: ' + data_path)

    #for d in data_dirs:
    #    d = os.path.join(data_path, d)
    #    #for fname in os.listdir(d):
    #    #path = os.path.join(d, fname)
    #    if os.path.isfile(d) == True:
    #        os.remove(d)
    print("Cleanup finished")
    print()
    #time.sleep(1) # wait to make sure there is no issue with copying into data



def run_tests(tests):
    cleanup("start")

    try:
        for test in tests:
            try:
                for name, t, args, kwargs in test:
                    print("Running", name, end=" ")
                    print("")
                    t(*args, **kwargs)
                    """
                    print()
                    print( len(document_file_list("partial")))
                    print( len(document_file_list("complete")))
                    print( len(document_file_list("2nd_annotation")))
                    print( len(document_file_list("3rd_annotation")))
                    print()
                    """
                    print("["+bcolors.OKGREEN+"OK"+bcolors.ENDC+"]")
            except Exception as e:
                print("["+bcolors.FAIL+"FAIL"+bcolors.ENDC+"]")
                traceback.print_exc()
            if NO_CLEANUP_AFTER:
                print("no cleanup performed")
            else:
                cleanup("end")
    except Exception as e:
        traceback.print_exc()
    finally:
        if NO_CLEANUP_AFTER:
            print("no cleanup performed")
        else:
            cleanup("end")

#def document_file_list(collection):
#    dir_path = os.path.join(data_path, collection)
#    filelist = os.listdir(dir_path)
#    filelist = [fname for fname in filelist if fname.endswith(".json")]
#    return filelist

################################
## TEST DEFINITIONS
################################

def get_statistics():
    get_json(base_url+"/get_annotation_session_statistics/")
    
def initial_queue_init():
    post_json({}, base_url+"/init_next_most_important/initial")

def queue_init(queue_type):
    post_json({}, base_url+"/init_next_most_important/" + queue_type)

def partial_queue_init():
    post_json({}, base_url+"/init_next_most_important/partial")

def initial_queue_run_simple():
    result = json.loads(get_json(base_url+"/next_most_important/"+annotator_ids[0]))
    print("#"*80)
    print("## next_most_important RESULT")
    print(result)
    document = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))
    print("#"*80)
    print("## DOCUMENT")
    print(document)

def initial_queue_run_full():
    print(); print('start: initial_queue_run_full')
    for i in range(7):

        get_result = get_json(base_url+"/next_most_important/"+annotator_ids[0])
        #print('try: get json: ' + base_url+"/next_most_important/"+annotator_ids[0])
        result = json.loads(get_result)
        #print('result["document_id"] = '); print(result["document_id"])
        
        document = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))
        assert document['initial_importance'] == 1 or document['initial_importance'] == 0
    for i in range(7):
        result = json.loads(get_json(base_url+"/next_most_important/"+annotator_ids[0]))
        document = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))
        assert document['initial_importance'] == 0 or document['initial_importance'] == 1

def partial_queue_run_full():
    for i in range(14):
        result = json.loads(get_json(base_url+"/next_most_important/"+annotator_ids[0]))
        print(result["importance_score"])
        document = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))
        

# posts files to server / stores in mongodb on server
# tag_to_add_to_filename ... used to change document_id to simulate having more documents in the database
check_if_docs_in_db = False # checks if added docs can be retrieved from db
def add_files_to_collection(test_collection, collection = None, num_docs = None, tag_to_add_to_document_id = None):
    print('\nstart: add_files_to_collection')
    print("num docs = " + str(num_docs) + " (note: if num docs is None all documents in collection are added)")
    test_folder = os.path.join(test_data, test_collection)
    print('test_folder: ' + test_folder)

    if check_if_docs_in_db:
        num_docs_in_collection_before = len(json.loads(get_json(base_url+"/get_all_documents/" + collection)))

    try:
        # save files that are currently in test_folder to database
        fnames_test = os.listdir(test_folder)
        fnames_test = set([fname for fname in fnames_test if fname.endswith(".json")])
        count = 0

        # put some files in priority and some in other
        for fname in fnames_test:

            path = os.path.join(test_folder, fname)
            with open(path, "r") as fp:
                document = json.load(fp)

            if collection != None: 
                document["target_collection"] = collection

            if tag_to_add_to_document_id != None:
                document["document_id"] = document["document_id"]+tag_to_add_to_document_id

            if (count == 0): #(count % 100)
                print('file nr. ' + str(count))
                print('post ' + path +' with ' + base_url+"/add_document/")
            
            result = post_json(document, base_url+"/add_document/")
            count += 1

            if num_docs != None and count == num_docs:
                break

        # assert that same number of files are stored in mongodb
        if check_if_docs_in_db:
            result = json.loads(get_json(base_url+"/get_all_documents/" + collection))

            print("added: " + str(count))
            print("retrieved from database: " + str(len(result)))
            #for i in result:
            #    print(i["document_id"])

            assert len(result)-num_docs_in_collection_before == count
    except FileNotFoundError:
        print("ERROR: test_data does not have folder: " + str(test_collection))


# posts files to server / stores in mongodb on server
def add_files_to_different_collections(test_collection, num_docs: None):
    print('\nstart: add_files_with_status')
    test_folder = os.path.join(test_data, test_collection)
    print('test_folder: ' + test_folder)

    # save files that are currently in test_folder to database
    fnames_test = os.listdir(test_folder)
    fnames_test = set([fname for fname in fnames_test if fname.endswith(".json")])
    count = 0

    # put some files in priority and some in other
    for fname in fnames_test:

        path = os.path.join(test_folder, fname)
        with open(path, "r") as fp:
            document = json.load(fp)
        if count < num_docs/3:
            document["target_collection"] = "priority"
        elif count < (num_docs/3)*2:
            document["target_collection"] = "other"
        elif count < num_docs:
            document["target_collection"] = "partial"

        if (count == 0): # % 50
            print('file nr. ' + str(count))
            print('post ' + path +' with ' + base_url+"/add_document/")
        
        result = post_json(document ,base_url+"/add_document/")
        count += 1

        if num_docs != None and count == num_docs:
            break

def copy_files(test_collection, collection):
    test_folder = os.path.join(test_data, test_collection)
    data_folder = os.path.join(data_path, collection)
    shutil.copytree(test_folder, data_folder)
    print(os.listdir(data_folder))

#def init_initial():
#    for collection in ["priority", "other"]:
#        src = os.path.join(test_data, collection)
#        dst = os.path.join(data_path, collection)
#        for fname in os.listdir(src):
#            src_path = os.path.join(src, fname)
#            dst_path = os.path.join(dst, fname)
#            shutil.copyfile(src_path, dst_path)
    

def main_queue_run():

    print('\npost json: ' + base_url+"/init_next_most_important/main")
    post_json({}, base_url+"/init_next_most_important/main")
    # first take 10 from complete -> 2nd annotation
    # annotator_0 should be available => not with new data

    # we get the next_most_important document and set its target_collection
    # to the target collection (target) specified by the queue 

    for i in range(35):
        result = json.loads(get_json(base_url+"/next_most_important/"+annotator_ids[0]))
        #print(result)

        if not isinstance(result, str):

            document = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))
            previous_target_collection = document["target_collection"] 
            document["target_collection"] = result["target"] 
            post_json(document, base_url+"/add_document/")
            document_updated = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))

            assert(document_updated["target_collection"] == result["target"])
            print(previous_target_collection + " -> " + str(document_updated["target_collection"]))

        else:
            #print(i)
            print("result is not a json: " + str(result))
            assert isinstance(result, str) 


def main_queue_run_until_15_percent_2nd_annotation():
    # partial: 61
    # complete: 85
    # 2nd_annotation: 15
    # 3rd_annotation: 25
    # first take 10 from complete -> 2nd annotation
    # annotator_0 should be available
    for i in range(15):
        result = json.loads(get_json(base_url+"/next_most_important/"+annotator_ids[0]))
        document = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))
        document['target_collection'] = result['target']
        result = post_json(document, base_url+"/add_document/")

    assert json.loads(get_json(base_url+"/get_number_of_documents/partial"))["num"] == 61  
    assert json.loads(get_json(base_url+"/get_number_of_documents/complete"))["num"] == 85
    assert json.loads(get_json(base_url+"/get_number_of_documents/2nd_annotation"))["num"]  == 15
    assert json.loads(get_json(base_url+"/get_number_of_documents/3rd_annotation"))["num"]  == 25


def total_queue_run():
    print('\nstart total_queue_run')

    post_json({}, base_url+"/init_next_most_important/total_priority")
    
    for i in range(10):
        result = json.loads(get_json(base_url+"/next_most_important/"+annotator_ids[0]))

        print("importance_score " + str(result["importance_score"]))

        document = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))
        previous_target_collection = document["target_collection"] 
        print("previous_target_collection " + previous_target_collection + " / complete annotation by " + str(len(document["annotated_by"]["annotators_complete"])) + " annotators")
        document['target_collection'] = result['target']
        post_json(document, base_url+"/add_document/")
        document_updated = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))
        assert document_updated["target_collection"] == result['target']
    
    #result = json.loads(get_json(base_url+"/next_most_important/"+annotator_ids[0]))
    #assert result == "Starting queue: total"



def ml_queue_run():
    print('\nstart: ml_queue_run')

    print('try connect to (POST request): ' + base_url+"/init_next_most_important/al_entropy")

    post_json({}, base_url+"/init_next_most_important/al_entropy")
    
    documents = []
    scores = []
    for i in range(10):        
        result = json.loads(get_json(base_url+"/next_most_important/"+annotator_ids[0]))
        
        document = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))
        document['target_collection'] = result['target']
        documents.append(result)
        scores.append(result['importance_score'])

        result = post_json(document, base_url+"/add_document/")

    #print(scores)
    
    # check if documents are returned sorted by importance_score
    assert sorted(scores, reverse=True) == scores

def start_pipeline ():
    print('\nstart start_pipeline')
    print('try connect to (POST request): ' + base_url+"/start_pipeline/")
    post_json({}, base_url+"/start_pipeline/")
    print('post successful')

def start_prediction ():
    post_json({}, base_url+"/start_prediction/")
    print("")

# save (some) documents from test_collection in mongodb
# this stores files locally not on the server
def store_documents_to_mongodb(test_collection):
    test_folder = os.path.join(test_data, test_collection)
    print('\ntest_folder: ' + test_folder)

    mongodb = MongoDB(None, None)
    
    fnames_test = os.listdir(test_folder)
    fnames_test = set([fname for fname in fnames_test if fname.endswith(".json")])

    count = 0
    for fname in fnames_test:

        path = os.path.join(test_folder, fname)
        with open(path, "r") as fp:

            # load file: filepath -> json
            document = json.load(fp) 

            document["target_collection"] = "initial"

            # save one document to mongodb (one document = one row)
            mongodb.save_to_collection(document) 

            print("save document with target_collection = " + document["target_collection"])

        count = count + 1
        if count == 50:
            break

def test_mongodb(test_collection):
    test_folder = os.path.join(test_data, test_collection)
    print('\ntest_folder: ' + test_folder)

    mongodb = MongoDB(None, None)

    jsondatastore = JSONDatastore([], {"testing": {"path":test_data}})
    queue_manager = QueueManager(jsondatastore, [])
    
    fnames_test = os.listdir(test_folder)
    fnames_test = set([fname for fname in fnames_test if fname.endswith(".json")])

    count = 0
    for fname in fnames_test:

        path = os.path.join(test_folder, fname)
        with open(path, "r") as fp:

            # load file: filepath -> json
            document = json.load(fp) 

            # save one document to mongodb (one document = one row)
            document["target_collection"] = "ml_out"
            mongodb.save_to_collection(document) 
            #print("save document " + str( document["document_id"]))

            # retrieve from datastore
            #mongodb.load_from_datastore(document["document_id"]) 

            # json -> Document
            document_obj = Document.deserialize(document) 
            #print(document_obj.document_id)

            jsondatastore.save_to_collection_without_score(document_obj, "testing") 

        count = count + 1
        #if count == 50:
        #    break


    # stop time to create generator and iterate -> 1000 - 1300 files / sec
    t_start = time.time() 
    docs = mongodb.document_generator("ml_out")
    
    c = 0
    for doc in docs:
        #print(doc.document_id)
        c = c + 1

    t_end = time.time()

    fps_mongo = c / (t_end - t_start) 
    print('--- mongodb ---')
    print('seconds passed to iterate all files ' + str((t_end - t_start)))
    print('documents found ' + str(c))
    print("Files per second: {:.2f}".format(fps_mongo)) 

    print('--- json datastore ---')

    t_start = time.time() 
    docs = jsondatastore.document_generator("testing")
    
    c_json = 0
    for doc in docs:
        c_json = c_json + 1

    t_end = time.time()

    # assert that mongodb and jsondatastore found the same number of documents
    assert c == c_json

    fps_jsondatastore = c_json / (t_end - t_start) 
    print('seconds passed to iterate all files ' + str((t_end - t_start)))
    print('documents found ' + str(c_json))
    print("Files per second: {:.2f}".format(fps_jsondatastore)) 

    print("Iterating over Documents with MongoDB is " + str(fps_mongo/fps_jsondatastore) + " x faster than with JSONDatastore") 


    
# uses mongodb
def basic_queue_generation():
    
    documents = []
    for i in range(2):
        print('\ntest: load next_most_important')
        result = json.loads(get_json(base_url+"/next_most_important/"+annotator_ids[0]))
        print('result = '); print(result)
        print('try load document id = ' + result["document_id"])


# uses mongodb
def score_based_queue_generation():
    
    documents = []
    for i in range(2):
        result = json.loads(get_json(base_url+"/next_most_important/"+annotator_ids[0]))
        print('result = '); print(result)
        print('try load document id = ' + result["document_id"])
        
    
# TODO test with ml_out data
# if there is only human annotation entropy score is 0 (because human annotation is one-hot-vector (such as [0,0,1]))
def test_queue_status():

    print("")
    t_start = time.time() 

    ### test setup ###
    # add some documents with status partial and complete
    # partial should be handled first, then other, then complete docs with annotation, 
    # then other with annotation, annotators should not see documents twice during the same annotation phase
    # assign letter to documents for easier reading
    # remove exiting annotators
    queue_init("al_entropy")  

    print("get content of queue. note: map document_ids to A, B, C,... for easier reading")
    document_keys = {}

    letters = ["A","B","C","D","E","F"]
    for i in range(6):
        result = json.loads(get_json(base_url+"/next_most_important/"+annotator_ids[0]))
        #if(result == None):
        #    print("no next_most_important found")
        #else:
        #    print("found: " + result["document_id"])

        document_keys[result["document_id"]] = letters[i]

        # ---- test setup: remove existing annotators -----
        document = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))

        document["annotated_by"]["annotators_partial"] = []
        document["annotated_by"]["annotators_complete"] = []

        for category in ["document_contains", "document_type", "document_label"]:
            document["annotated_by"][category] = []

        for category in ["is_occluded", "segment_boundary", "segment_label", "segment_type"]:
            document["annotated_by"][category] = []

        post_json(document, base_url+"/add_document/")
        # --------------------------------------------------


    for x in document_keys:
        print(x + " -> " + document_keys[x])

    queue_init("al_entropy")
    print()

    doc_list = []

    # partial annotation -> complete annotation -> 2nd complete annotation -> 3rd complete annotation
    annotators = [1,2,1,0,1,1, 1,2,1,2,2,1, 1,1,2,1,0,0, 0,0,1,2]

    for current_annotator in annotators:  
        #print("get document for: " + str(annotator_ids[current_annotator]))
        result = json.loads(get_json(base_url+"/next_most_important/"+annotator_ids[current_annotator]))

        if (result != None):
            # add annotation by current annotator
            #post_json({}, base_url+"/add_annotation/"+ result["document_id"] +"/"+ annotator_ids[current_annotator] + "/")
            # check for annotation
            
            add_annotation(result["document_id"], result["target"], current_annotator)

            doc = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))
            #print("after annotation")
            #print(str(document_keys[result["document_id"]]) + ": " 
            #+ " target_collection:" + str(doc["target_collection"])
            #+ " score:" + str(doc["importance_score"]))
            #print(doc["annotated_by"]["annotators_partial"])
            #print(doc["annotated_by"]["annotators_complete"])
            #print()
            doc_list.append(document_keys[result["document_id"]])
        else:
            #print("no documents left for this annotator \n")
            doc_list.append('')
            
    print(doc_list)
    t_end = time.time()
    print("seconds passed per item in queue: " +str((t_end - t_start)/len(annotators)))
    assert(doc_list == ['A', 'B', 'C', 'D', 'E', 'F', 'A', 'B', 'C', 'D', 'E', 'F', 'B', 'D', 'A','E','C','F', 'A', 'B', '', 'C' ])



def test_large_queue():

    # priority should be handled first, then other, then priority docs with annotation, then other with annotation
    print("")
    t_start = time.time()
    queue_init("initial") 

    doc_list = []

    annotators =  [1,2,1,2,2,1, 1,1,2,1,0,0, 0,0,1,2]

    for current_annotator in annotators:  
        result = json.loads(get_json(base_url+"/next_most_important/"+annotator_ids[current_annotator]))

        #print("annotator: " + str(annotator_ids[current_annotator]))

        doc = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))  
        #print("target_collection before annotation: " + doc["target_collection"])

        add_annotation(result["document_id"], result["target"], current_annotator)

        if (result != None):
            # add annotation by current annotator

            doc = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))  

            #print("score: " + str(doc["importance_score"]))
            #print("target_collection: " + doc["target_collection"])
            #print("annotated partial: " + str(doc["annotated_by"]["annotators_partial"]))
            #print("annotated complete: " + str(doc["annotated_by"]["annotators_complete"]))
            #print()

            doc_list.append(doc["document_id"])
        else:
            print("no documents left for this annotator \n")
            doc_list.append('')


    t_end = time.time()
    print("seconds passed per item from larger queue: " + str((t_end - t_start)/len(annotators)))
    #print(doc_list)


def add_annotation(doc_id, target, current_annotator):
    doc = json.loads(get_json(base_url+"/get_document/"+doc_id))  

    #print("target_collection before annotation: " + doc["target_collection"])

    #  add annotated_by and store document in database

    if (doc["target_collection"] == "other" or doc["target_collection"] == "priority"):
        doc["annotated_by"]["document_contains"].append({"name": annotator_ids[current_annotator]}) # add partial annotation
        doc["target_collection"] = target #"partial"

    elif (doc["target_collection"] == "partial"):
        doc["annotated_by"]["segment_boundary"].append({"name": annotator_ids[current_annotator]}) # add complete annotation
        doc["target_collection"] = target #"complete"

    elif (doc["target_collection"] == "complete"):
        doc["annotated_by"]["segment_boundary"].append({"name": annotator_ids[current_annotator]}) # add complete annotation
        doc["target_collection"] = target # complete

    elif (doc["target_collection"] == "ml_out"):
        doc["annotated_by"]["document_contains"].append({"name": annotator_ids[current_annotator]}) # add partial annotation?
        doc["target_collection"] = target #"partial"
        
    post_json(doc, base_url+"/add_document/")



def test_queue_condition():

    #{"$cond":[{"$eq":X}, 1,
    #    {"$cond":[{"$eq":Y}, 2]} 
    #]}

    collection_order = ["priority", "other", "partial", "complete"]

    expected = {"$cond":[{"$eq":["$target_collection", collection_order[0]]}, 1,
        {"$cond":[{"$eq":["$target_collection", collection_order[1]]}, 2,  
            {"$cond":[{"$eq":["$target_collection", collection_order[2]]}, 3,  
                {"$cond":[{"$eq":["$target_collection", collection_order[3]]}, 4,  
                    5
                ]} 
            ]} 
        ]} 
    ]}

    complete_wrapper = {}
    wrapper = {}
    conditition = [{"$eq":["$target_collection", collection_order[0]]}, 1]
    complete_wrapper["$cond"] = conditition

    for i in range(1, len(collection_order)):
        wrapper = {}
        conditition.append(wrapper)
        conditition = [{"$eq":["$target_collection", collection_order[i]]}, i+1]

        if i == len(collection_order)-1:
            conditition.append(i+2)

        wrapper["$cond"] = conditition

    if len(collection_order) == 1:
        conditition.append(2)


    #print("\ncondition" + str(conditition))
    #print("wrapper" + str(wrapper))
    #print("complete_wrapper" + str(complete_wrapper))

    assert complete_wrapper == expected


# rough "evaluation"
# check if different scoring functions deliver different but similar results
def test_scoring_functions():
    print("")

    annotators =  [1,1,1,1,1]
    queue_types = ["al_entropy", "al_least_confident", "al_margin_sampling"]
    doc_list = []

    for queue_type in queue_types:
        queue_init(queue_type) 
        print("queue type = " + queue_type)

        sub_list = []

        for current_annotator in annotators:  
            result = json.loads(get_json(base_url+"/next_most_important/"+annotator_ids[current_annotator]))
            if (result != None):
                doc = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))  
                sub_list.append(doc["document_id"] + " " + str(doc["importance_score"]))
            else:
                print("no documents left for this annotator \n")
                sub_list.append('')

        doc_list.append(sub_list)
            
    print("")  
    for i in range(len(annotators)):
        for j in range(len(queue_types)):
            print(doc_list[j][i] + " " + queue_types[j])
        print("")

    #print(doc_list)

# estimate how much time the AL component needs
def measure_runtime():
    # setup: added 5000 / 50.000 documents

    # create queue
    t_start = time.time() 
    queue_init("al_entropy")
    t_end = time.time()
    print("\ninitializing queue")
    print(t_end - t_start)

    # retrieve doc
    t_start = time.time() 
    annotators = [1,1,1,1,1, 2,2,2,2,2]
    for current_annotator in annotators:  
        result = json.loads(get_json(base_url+"/next_most_important/"+annotator_ids[current_annotator]))

    t_end = time.time()
    print("\nretrieve one document:")
    print((t_end - t_start) / len(annotators))

    # update doc
    t_start = time.time() 
    for current_annotator in range(10):  
        document = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))
        post_json(document, base_url+"/add_document/")

    t_end = time.time()
    print("\nadd one document:")
    print((t_end - t_start) / 10)
    

def save_to_folder():
    # give path to folder as parameter (json files can be in top level of folder or inside collection folders); 
    # path must be reached from database folder 
    # surround path with single quotes (') if path contains slashes  
    post_json({}, base_url+"/add_documents_in_folder/'copy_files_to_db/test_files'")


def complete_queue_reset():
    post_json({}, base_url+"/init_next_most_important/complete")
    for i in range(21):
        result = json.loads(get_json(base_url+"/next_most_important/"+annotator_ids[0]))
    for i in range(21):
        result = json.loads(get_json(base_url+"/next_most_important/"+annotator_ids[0]))


def add_many_docs(num_docs, batch_size, from_target, to_target):

    iterations = (int) (num_docs / batch_size)
    rest = num_docs % batch_size

    for i in range(iterations): 
        print("\nadd: " + str((i+1)*batch_size))
        add_files_to_collection(from_target, to_target, batch_size, "_tag_"+str(i))

    if rest > 0:
        print("add rest" + str(rest))
        add_files_to_collection(from_target, to_target, rest, "_tag_"+ "X")


# get human annotations from document 
def get_human_annotation_basic():

    # get a document
    document = json.loads(get_json(base_url+"/get_document/000dfd749a544371af076f6d80a448db"))

    # convert to Document object
    document_obj = Document.deserialize(document) 
    for annotation in document_obj.annotations():
        print("")
        print(annotation.annotation_id)
        print(annotation.annotator_id)
        print(annotation.annotator_type)
        print(annotation.annotation_vector)
        print(annotation.annotation_type)

def get_human_annotation_missing_file():

    # 1. file exists
    document = json.loads(get_json(base_url+"/get_human_annotations/000dfd749a544371af076f6d80a448db"))
    print(document)

    # 2. file does not exist
    result = get_json(base_url+"/get_human_annotations/123")
    print(result)




def test_ml_out_diversity():

    print("")
    t_start = time.time() 

    ### test setup ###
    # add some documents with status partial and complete
    # partial should be handled first, then other, then complete docs with annotation, 
    # then other with annotation, annotators should not see documents twice during the same annotation phase
    # assign letter to documents for easier reading
    # remove exiting annotators
    queue_init("ml_out_diversity")

    print("get content of queue. note: map document_ids to A, B, C,... for easier reading")
    document_keys = {}

    letters = ["A","B","C","D","E","F"]
    for i in range(6):
        result = json.loads(get_json(base_url+"/next_most_important/"+annotator_ids[0]))

        document_keys[result["document_id"]] = letters[i]

        # ---- test setup: remove existing annotators -----
        document = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))

        document["annotated_by"]["annotators_partial"] = []
        document["annotated_by"]["annotators_complete"] = []

        for category in ["document_contains", "document_type", "document_label"]:
            document["annotated_by"][category] = []

        for category in ["is_occluded", "segment_boundary", "segment_label", "segment_type"]:
            document["annotated_by"][category] = []

        post_json(document, base_url+"/add_document/")
        # --------------------------------------------------


    for x in document_keys:
        print(x + " -> " + document_keys[x])

    queue_init("ml_out_diversity")
    print()

    doc_list = []

    annotators = [1,1,1,1,1,1] #[1,2,1,1,1,1, 1,1,1,1,1,1, 2,2,2,1,1,1]

    for current_annotator in annotators:  
        #print("get document for: " + str(annotator_ids[current_annotator]))
        result = json.loads(get_json(base_url+"/next_most_important/"+annotator_ids[current_annotator]))

        if (result != None):
            
            add_annotation(result["document_id"], result["target"], current_annotator)

            doc = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))
            
            #print("after annotation")
            print(str(document_keys[result["document_id"]]) + ": " 
            + " target_collection:" + str(doc["target_collection"]) + "\n"
            + "score:" + str(doc["importance_score"]))
            #print(doc["annotated_by"]["annotators_partial"])
            #print(doc["annotated_by"]["annotators_complete"])
            #print()
            doc_list.append(document_keys[result["document_id"]])
        else:
            #print("no documents left for this annotator \n")
            doc_list.append('')
            
    print(doc_list)
    t_end = time.time()
    print("seconds passed per item in queue: " +str((t_end - t_start)/len(annotators)))
    #assert(doc_list == ['A', 'B', 'C', 'D', 'E', 'F', 'A', 'B', 'C', 'D', 'E', 'F', 'B', 'D', 'A','E','C','F', 'A', 'B', '', 'C' ])


def human_annotations():

        post_json({}, base_url+"/init_next_most_important/complete")

        for i in range(50):
            doc = json.loads(get_json(base_url+"/next_most_important/"+annotator_ids[0]))
            annotations = json.loads(get_json(base_url+"/get_human_annotations/"+doc["document_id"]))
            for a in annotations["annotations"]:
                #print("")
                #print(a["annotation_type"])
                #print(a["annotation_id"])
                #print(a["annotation_vector"])
                #print(a["segment_id"])
                if a["page_number"]!= "" and a["page_number"] > 0:
                    print(a["page_number"])


def split_dataset():
    post_json({}, base_url+"/split_dataset/1/1/1")


def store_db_to_folders():
    post_json({}, base_url+"/store_db_to_folders/")

def test_store_db_to_folders():
    post_json({}, base_url+"/store_db_to_folders/")

    queue_init("al_partial")
    next = json.loads(get_json(base_url+"/next_most_important/X"))
    next = json.loads(get_json(base_url+"/get_document/"+ next["document_id"]))

    old_target_collection = next["target_collection"] 
    next["target_collection"] = "new_collection"

    post_json(next, base_url+"/add_document/")

    
    post_json({}, base_url+"/store_db_to_folders/")

    print("current dir ", os.getcwd())
    data_path = os.path.join(os.getcwd(), 'mongodb')
    data_dirs = os.listdir(data_path)

    # create a file that is not in db
    print("data_path " + data_path)

    print("make directory " + os.path.join(data_path, "not-in-db"))
    if os.path.isdir(os.path.join(data_path, "not-in-db")) == False:
        os.mkdir(os.path.join(data_path, "not-in-db"))
    f = open(os.path.join(data_path, "not-in-db","not-in-db.json"), "w")
    f.close()

    d = dict()
    for data_dir in data_dirs:
        counter = 0
        data_dir = os.path.join(data_path, data_dir)
        if os.path.isdir(data_dir):

            for fname in os.listdir(data_dir):
                file_path = os.path.join(data_dir, fname)

                doc_id = fname[:len(fname)-len(".json")]
                doc = json.loads(get_json(base_url+"/get_document/"+doc_id))

                #remove doc if: 
                # doc is in db 
                # and target collection of doc in db is not same as foldername
                if (os.path.isfile(file_path) == True
                    and doc != {} 
                    and doc["target_collection"] != data_dir):
                    counter = counter + 1
            
            d[data_dir] = counter


    # check: file is in folder but not in db -> should stay
    assert(os.path.isfile(os.path.join(data_path, "not-in-db/not-in-db.json")))

    # check: file is stored then target collection changes 
    # -> should be removed from old folder and added to new folder
    assert(os.path.isfile(os.path.join(data_path, old_target_collection + "/" + next["document_id"]+".json")) == False)
    assert(os.path.isfile(os.path.join(data_path, "new_collection/"+next["document_id"]+".json")))

    #clean up
    os.remove(os.path.join(data_path, "not-in-db/not-in-db.json"))
    os.rmdir(os.path.join(data_path, "not-in-db"))

    for data_dir in data_dirs:
        data_dir = os.path.join(data_path, data_dir)
        if os.path.isdir(data_dir):

            for fname in os.listdir(data_dir):
                os.remove(os.path.join(data_path, data_dir, fname))
            
            os.rmdir(os.path.join(data_path, data_dir))

    print(d)


def run_partial_queue():
    post_json({}, base_url+"/init_next_most_important/al_partial")

    result = ""
    while result != None:  
        result = json.loads(get_json(base_url+"/next_most_important/X"))
        print(result)
        if result != None: 
            doc = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))
            # set target
            doc["target_collection"] = result["target"]
            post_json(doc, base_url+"/add_document/")


def try_to_get_invalid_id():

    doc = json.loads(get_json(base_url+"/get_document/468b5e1020eb2f05825fcedc1ec7380a_ml"))
    print(doc["document_id"])

    doc = get_json(base_url+"/get_document/123")
    if doc == None:
        print("is none")

    print(doc)


def no_repeat_in_partial():
    viewed_docs = []

    # first view 100 documents
    post_json({}, base_url+"/init_next_most_important/al_partial")

    result = ""
    #while result != None: 
    for i in range(100):
        result = json.loads(get_json(base_url+"/next_most_important/"+ str(random.randint(0,9))))
        if  result != None:
            viewed_docs.append(result["document_id"])
            
            doc = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))

            # do not set target user has doc open
            # set target!
            #doc["target_collection"] = result["target"]
            #post_json(doc, base_url+"/add_document/")
            #total_num_docs += 1
    
    # TEST no repeat
    # check that docs are not repeated in the viewing queue
    for idx, viewed in enumerate(viewed_docs):
        assert([i for i, e in enumerate(viewed_docs) if e == viewed] == [idx])


    print("number of viewed docs " + str(len(viewed_docs)))

    # test rest call
    result = json.loads(get_json(base_url+"/get_blocked_ids/"))
    print("blocked ids: " + str(len(result)))




def test_annotation_session():

    viewed_docs = []
    partial_docs = []
    complete_docs = []

    #total_num_docs = 

    # first view 100 documents
    post_json({}, base_url+"/init_next_most_important/al_view")

    result = ""
    #while result != None: 
    for i in range(10):
        result = json.loads(get_json(base_url+"/next_most_important/"+ str(random.randint(0,9))))
        if  result != None:
            viewed_docs.append(result["document_id"])
            
            doc = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))

            # set target!
            doc["target_collection"] = result["target"]
            post_json(doc, base_url+"/add_document/")
            #total_num_docs += 1

    #print("total_num_docs " + str(total_num_docs))


    post_json({}, base_url+"/reset_target/viewed/ml_out")  # done automatically but can be done to reset view queue
    time.sleep(1)

    # check that docs are not repeated in the viewing queue
    for idx, viewed in enumerate(viewed_docs):
        assert([i for i, e in enumerate(viewed_docs) if e == viewed] == [idx])

    print("view queue finished")

    # now annotate <5000> partial and <2500+> complete
    post_json({}, base_url+"/init_next_most_important/al_partial")

    count_add_to_partial = 0
    for i in range(1, 60):
        #print(i)
        result = json.loads(get_json(base_url+"/next_most_important/"+ str(random.randint(0,9))))

        # set target - but skip sometimes!
        skip_docs= [10, 15, 20]
        if i not in skip_docs:
            doc = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))
            doc["target_collection"] = result["target"]
            post_json(doc, base_url+"/add_document/")

        if i == config["debug"]["DEBUG_switch_to_partial_after_X_documents"] + len(skip_docs):
            print("reached switch value", i)
            print(result["queue"])
            num_of_docs = json.loads(get_json(base_url+'/get_number_of_documents_for_all_collections')) # check that we have correct num of docs
            print(num_of_docs)

        #print (i)
        #num_of_docs = json.loads(get_json(base_url+'/get_number_of_documents_for_all_collections')) # check that we have correct num of docs
        #print(num_of_docs)

        if i > config["debug"]["DEBUG_switch_to_partial_after_X_documents"] + len(skip_docs):  
            #if i % 10 == 0:
            #print(i, " ", result["queue"])
            assert(result["queue"] == "al_complete")
            complete_docs.append(result["document_id"])
        else:
            #if i % 10 == 0:
            #print(i, " ", result["queue"])
            if i not in skip_docs:
                count_add_to_partial +=1 
                assert(result["queue"] == "al_partial")
                partial_docs.append(result["document_id"])

    print("seen partial docs ",  len(partial_docs))
    print("count_add_to_partial ",  count_add_to_partial)

    num_of_docs = json.loads(get_json(base_url+'/get_number_of_documents_for_all_collections')) # check that we have correct num of docs
    for n in num_of_docs:
        if n["target_collection"] == 'has_partial_human_annotation':
            assert(n["num"] == 50)

    assert(len(partial_docs) == config["debug"]["DEBUG_switch_to_partial_after_X_documents"])
            
    #result = json.loads(get_json(base_url+"/next_most_important/"+ str(random.randint(0,9))))
    #assert(result == None)


    # check that docs in view queue are shown again (exactly once) in partial queue
    for idx, viewed in enumerate(viewed_docs):
        assert(len([i for i, e in enumerate(viewed_docs) if e == viewed]) == 1)

    # check that docs in partial queue are not shown in complete queue
    for idx, partial in enumerate(partial_docs):
        assert(len([i for i, e in enumerate(complete_docs) if e == partial]) == 0)

        
def test_evaluation():
    #post_json({}, base_url+"/start_prediction/")

    #test_annotation_session() TODO we need documents with ml output and human annotation to test properly

    #post_json({}, base_url+"/split_dataset/")
    #post_json({}, base_url+"/start_evaluation/")


    ### run wit individual rest calls ###
    print(get_json(base_url+"/get_number_of_documents_for_all_collections"))
    print(get_json(base_url+"/get_document_label_distribution/"))

    post_json({}, base_url+"/split_dataset/1/1/1")

    print("split done")


    post_json({}, base_url+"/start_base_predictions/")
    post_json({}, base_url+"/start_finetuning_training/")
    post_json({}, base_url+"/start_active_learning/efficient_annotation/")
    post_json({}, base_url+"/start_active_learning/random/")

    print("finished evalutation")
    result1 = json.loads(get_json(base_url+"/get_dataset_split/"))
    #print(result)

    print("")

    ### run with one wrapper rest call ###
    # should be the same
    post_json({}, base_url+"/start_evaluation/1/1/1")
    print("finished evalutation")
    result2 = json.loads(get_json(base_url+"/get_dataset_split/"))

    print(get_json(base_url+"/get_number_of_documents_for_all_collections"))
    print(get_json(base_url+"/get_document_label_distribution/"))

    print(result1['initial_finetuning_batch'])


    #assert(result1['random_finetuning_batches'] == result2['random_finetuning_batches'])
    assert(result1['al_finetuning_batches']['ids'] == result2['al_finetuning_batches']['ids'])
    assert(result1['initial_finetuning_batch']['ids'] == result2['initial_finetuning_batch']['ids'])


def test_evaluation_too_small():
    post_json({}, base_url+"/split_dataset/10/10")
    post_json({}, base_url+"/start_base_predictions/")
    post_json({}, base_url+"/start_finetuning_training/")
    post_json({}, base_url+"/start_active_learning/efficient_annotation/")
    post_json({}, base_url+"/start_active_learning/random/")


def test_filtered_finetuned_annotation_types():

    result = json.loads(get_json(base_url+"/get_filtered_finetuned_annotation_types/"))
    print(result)


def test_eval_random():
    print("")
    print("eval_random")
    post_json({}, base_url+"/init_next_most_important/eval_random")
    for i in range(20):
        result = json.loads(get_json(base_url+"/next_most_important/"+"X"))
        print(result)

    print("")
    print("eval_al")

    post_json({}, base_url+"/init_next_most_important/eval_al")
    for i in range(20):
        result = json.loads(get_json(base_url+"/next_most_important/"+"X"))
        print(result)

def test_diversity_queue_in_db():
    mongodb = MongoDB(None, None, "mongodb://localhost:27017/")
    
    for i in range(19): # 19 = no. document label classes
        print("i", i)

        document_json = mongodb.get_document_with_predicted_document_label("ml_out", i)
        if (document_json != None):
            for document_label_annotation in document_json["document_label"]["annotations"]:
                if document_label_annotation["annotator_id"] == "segment_multimodal_model":
                    vec = document_label_annotation["annotation_vector"]
                    maximum = max(vec)
                    index = [i for i, j in enumerate(vec) if j == maximum] 
                    
                    print("vec", vec)
                    print("maximum", maximum)
                    print("index", index)



def test_diversity_queue_small():
    post_json({}, base_url+"/init_next_most_important/al_partial")
    
    for i in range(10):  
        result = json.loads(get_json(base_url+"/next_most_important/"+"X"))
        #doc = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))  
        print("result", result)
        assert(result["queue"] == "al_partial")

    
def empty_log():
    for filename in ["accuracy.txt", "log.txt", "queue_log.txt" ]:
        post_json({}, base_url+"/empty_log/" + filename)

    
def test_diversity_queue():
    post_json({}, base_url+"/init_next_most_important/al_partial")
    
    start_time = time.time()
    print("for this test config.js debug>DEBUG_switch_to_partial_after_X_documents should be set to 5000 is ->" + str(config["debug"]["DEBUG_switch_to_partial_after_X_documents"]))
    for i in range(5000):  
        result = json.loads(get_json(base_url+"/next_most_important/"+"X"))
        
        # set target
        doc = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))
        doc["target_collection"] = result["target"]
        post_json(doc, base_url+"/add_document/")
        #print("result", result)

        assert(result["queue"] == "al_partial")

    
    #post_json({}, base_url+"/init_next_most_important/al_complete")
    for i in range(2500):  
        result = json.loads(get_json(base_url+"/next_most_important/"+"X"))

        # set target
        doc = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))
        doc["target_collection"] = result["target"]
        post_json(doc, base_url+"/add_document/")
        #print("result", result)

        assert(result["queue"] == "al_complete") # test automatic switch


    end_time = time.time()
    print("time elapsed:" + str(end_time-start_time))
    print("time per doc:" + str((end_time-start_time)/(5000+2500)))



def test_annotation_version():
    result = json.loads(get_json(base_url+"/get_ml_annotations/468b5e1020eb2f05825fcedc1ec7380a_ml"))
    
    print("ml:")
    #check that annotation_id test_version_1 is there and test_version_0 not
    for a in result["annotations"]:
        if a["annotation_type"] == "document_label" and a["annotator_id"] == "segment_multimodal_model" :
            print(a)

    print("\nhuman:")
    result = json.loads(get_json(base_url+"/get_human_annotations/468b5e1020eb2f05825fcedc1ec7380a_ml"))
    #check that annotation_id test_version_1 is there and test_version_0 not
    for a in result["annotations"]:
        if a["annotation_type"] == "document_label" :
            print(a)

    print("\n")
    for a in result["annotations"]:
        if a["annotation_type"] == "segment_label" and a["segment_id"] ==  "0n5V3VVr":
            print(a)
    


def call_rest_calls_in_document_annotator():
    # create a password manager
    password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()

    # Add the username and password.
    # If we knew the realm, we could use it instead of None.
    top_level_url = "https://documentannotator.k-rex.at"
    password_mgr.add_password(None, top_level_url, 'K.Rex', '187#tbFa')

    handler = urllib.request.HTTPBasicAuthHandler(password_mgr)

    # create "opener" (OpenerDirector instance)
    opener = urllib.request.build_opener(handler)

    # use the opener to fetch a URL
    opener.open("https://documentannotator.k-rex.at")

    # Install the opener.
    # Now all calls to urllib.request.urlopen use our opener.
    urllib.request.install_opener(opener)

    #result = json.loads(get_json("https://documentannotator.k-rex.at/api/activelearning/get_annotation_session_statistics/"))
    #print(result)
    
    #result = json.loads(get_json("https://documentannotator.k-rex.at/api/activelearning/start_prediction/"))
    #print(result)


def test_random():
    random.seed(0.5)
    print(random.random())
    print(random.random())
    print(random.random())
    print()

    random.seed(0.5)
    print(random.random())
    print(random.random())
    print(random.random())


def next_most_important_before_queue_init():
    # run next_most_important before init_next_most_important
    # should catch error and return "{}"
    # 
    result = json.loads(get_json(base_url+"/next_most_important/"+annotator_ids[0]))
    assert(json.loads(result) == {})


def get_number_of_docs():
    result1 = json.loads(get_json(base_url+"/get_number_of_documents/ml_out"))["num"]
    result2 = json.loads(get_json(base_url+"/get_number_of_documents/complete"))["num"]
    result3 = json.loads(get_json(base_url+"/get_number_of_documents"))["num"]
   
    print(result1)
    print(result2)
    print(result3)

    assert(result1 == 10) 
    assert(result2 == 20) 
    assert(result3 == 30) 


def test_al_view_finetuned():

    # test setup: set some docs to has_finetuned
    queue_init("al_view_finetuned")
    for i in range(20):  
        result = json.loads(get_json(base_url+"/next_most_important/"+"X"))
        doc = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))
        if i in [1,3,5,7,9,11,13,15,17,19]:
            doc["has_finetuned"] = True
            post_json(doc, base_url+"/add_document/")


    queue_init("al_view_finetuned")

    print(json.loads(get_json(base_url+"/get_number_of_documents_for_all_collections")))

    for i in range(45):  
        result = json.loads(get_json(base_url+"/next_most_important/"+"X"))
        #print(result)

        # set target
        doc = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))
        #doc["target_collection"] = result["target"]
        #post_json(doc, base_url+"/add_document/")
        #print("target_collection: ", doc["target_collection"])

        if i < 10:
            print(i, " has finetuned =", doc["has_finetuned"], " doc target_collection = ", doc["target_collection"])
            assert(doc["has_finetuned"] == True)
        if i >= 10 and i < 20:
            print(i, " has base =", doc["has_base"], " doc target_collection = ", doc["target_collection"])
            assert(doc["has_base"] == True)
        if i >= 20 and i < 30:
            print(i, " ", doc["target_collection"])
            assert(doc["target_collection"] == "ml_out")
        if i >= 30 and i < 40:
            print(i, " ", doc["target_collection"])
            assert(doc["target_collection"] == "partial")
        if i > 40 and i < 45:
            print(i, " ", doc["target_collection"])
            assert(doc["target_collection"] == "other")

        #if "has_finetuned" in doc:
        #    print(doc["has_finetuned"])
        #print("")

        #assert(result["queue"] == "al_complete") # test automatic switch

    # reset all has_finetuned to false
    queue_init("al_view_finetuned")
     
    while True:  
        result = json.loads(get_json(base_url+"/next_most_important/"+"X"))
        if result == None:
            break
        doc = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))
        doc["has_finetuned"] = False
        post_json(doc, base_url+"/add_document/")


def test_accuracy():
    queue_init("al_view_finetuned")
    for i in range(1):  
        result = json.loads(get_json(base_url+"/next_most_important/"+"X"))
        doc = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))
        for annotation in doc["document_label"]["annotations"]:
            if annotation["annotator_type"] == "human":
                print(annotation)


    # test base prediction scenario    
    post_json({}, base_url+"/split_dataset/1/1/1")
    post_json({}, base_url+"/start_base_predictions/")
    
    #TODO test finetuned with finetuned output 
    #post_json({}, base_url+"/start_finetuning_training/")
    #post_json({}, base_url+"/start_active_learning/efficient_annotation/")
    #post_json({}, base_url+"/start_active_learning/random/")


    
def test_start_finetuned_predictions():
    # post with empty
    print(post_json({}, base_url+"/start_finetuned_predictions/"))
    
    # post with payload
    assert("Sent 3 ids" in post_json(['21318f6e12c83acd98a0d1ab33a64617', 'c8488ffbe30a0ba7c6cdffdeb95bb475', '1f766caa049cf705174b1f1aa5baa85e'], base_url+"/start_finetuned_predictions/"))


def test_skipping_ids():
    # create a doc 55f96e15fdac51ccdbaa9a71c6a5edfe
    # should be skipped

    # create a doc 55f96e15fdac51ccdbaa9a71c6a5edfe
    # should not be skipped

    queue_init("al_view_finetuned")
    for i in range(2):  
        result = json.loads(get_json(base_url+"/next_most_important/"+"X"))
        doc = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))
        old_id = doc["_id"]
        doc["_id"] = "55f96e15fdac51ccdbaa9a71c6a5edfe"
        doc["document_id"] = "55f96e15fdac51ccdbaa9a71c6a5edfe"
        doc["has_base"] = True
        post_json(doc, base_url+"/add_document/")

        old_id2 = doc["_id"]
        doc["_id"] = "1e831451dd2ce3dbf36e528a5dbabbd9"
        doc["document_id"] = "1e831451dd2ce3dbf36e528a5dbabbd9"
        doc["has_base"] = False
        post_json(doc, base_url+"/add_document/")
    
    result = post_json({}, base_url+"/start_prediction/") 
    print(result)
    assert("Sent 8 ids" in result)

    #should predict only 1 doc
    print("start_finetuned_predictions")
    message = post_json({}, base_url+"/start_finetuned_predictions/55d01cd92044759281cfc431e57a07db")
    print(message)
    assert( "Sent 1 ids" in message)
    # should send only 1e831451dd2ce3dbf36e528a5dbabbd9
    message = post_json({}, base_url+"/start_finetuned_predictions/123")
    assert("has length 0" in message)

    # assert doc 55f96e15fdac51ccdbaa9a71c6a5edfe that has base predictions
    # is removed
    result = remove_docs_with_base_prediction(["55f96e15fdac51ccdbaa9a71c6a5edfe", "1e831451dd2ce3dbf36e528a5dbabbd9"])
    print(result)
    assert(result == ['1e831451dd2ce3dbf36e528a5dbabbd9'])


import copy
def remove_docs_with_base_prediction(payload):

    payload_copy = copy.deepcopy(payload)
    for doc_id in payload:

        result = get_json(base_url+"/get_document/"+doc_id)
        #simulate REST_AL
        if result != "{}":
            result = json.loads(result)
        
        print(result == {})
        print(result == "{}")

        if result != "{}":
            if "has_base" in result and result["has_base"] == True:
                payload_copy.remove(doc_id)
            elif "has_base" not in result:
                print("ERROR: json does not have key has_base")
    return payload_copy


def read_times_from_log():
    # read line by line
    # extract times 

    # if log file is longer: first move backwards to first occurance of EVALUATION

    path_to_log = r"C:\Users\anna_\Downloads\testrun-200925-2\log.txt"
  
    al_random_start = -1
    with open(path_to_log) as fp:
        line = fp.readline()
        cnt = 1
        while line:

            if "number of documents in target_collections has_complete_human_annotation and has_partial_human_annotation before removing documents" in line:
                total_docs = line.split("] ")[1]
                total_docs = total_docs.replace("number of documents in target_collections has_complete_human_annotation and has_partial_human_annotation before removing documents:","")
                total_docs = int(total_docs.strip())
                print("total number of documents with human annotation:", total_docs)

            if "number of documents after removing documents with zero-only human annotation and documents with" in line:
                filtered_docs = line.split("] ")[1]
                print(filtered_docs.replace("\n", ""))
                filtered_docs = filtered_docs.replace("number of documents after removing documents with zero-only human annotation and documents with","")
                filtered_docs = filtered_docs.split(":")[1]
                filtered_docs = int(filtered_docs.strip())

            # split
            if "split dataset with test_split" in line:
                time = line.split("] ")[0].replace("[time: ","")
                split_start = float(time)
                total_evaluation_start = float(time)

            if "first dataset split finished" in line:
                time = line.split("] ")[0].replace("[time: ","")
                split_end = float(time)
                print("time elapsed for dataset split: ", split_end-split_start, "s")   

            # base prediction
            if "start_base_predictions" in line:
                time = line.split("] ")[0].replace("[time: ","")
                base_start = float(time)

            if "Done: base_predictions" in line:
                time = line.split("] ")[0].replace("[time: ","")
                base_end = float(time)
                print("time elapsed for base_predictions: ", base_end-base_start, "s")


            # initial finetuning
            if "start_finetuning_training" in line:
                time = line.split("] ")[0].replace("[time: ","")
                initial_ft_start = float(time)

            if "start_active_learning with type efficient_annotation" in line:
                time = line.split("] ")[0].replace("[time: ","")
                initial_ft_end = float(time)
                print("time elapsed for initial finetuning: ", initial_ft_end-initial_ft_start, "s")


            # active learning: efficient annotation TODO
            if "start_active_learning with type efficient_annotation" in line:
                time = line.split("] ")[0].replace("[time: ","")
                al_efficient_start = float(time)

            if "start_active_learning with type random" in line:
                time = line.split("] ")[0].replace("[time: ","")
                al_efficient_end = float(time)
                print("time elapsed for active learning: efficient annotation: ", al_efficient_end-al_efficient_start, "s")


            # active learning: random
            if "start_active_learning with type random" in line:
                time = line.split("] ")[0].replace("[time: ","")
                al_random_start = float(time)

            if "[time: " in line:
                last_line_with_time = line


            line = fp.readline()            
            cnt += 1

        # at end of file: active learning: random
        time = last_line_with_time.split("] ")[0].replace("[time: ","")
        al_random_end = float(time)
        total_evaluation_end = float(time)
        print("time elapsed for active learning: random: ", al_random_end-al_random_start, "s")

        print("total time elapsed for evaluation: ", total_evaluation_end-total_evaluation_start, "s", " = ", (total_evaluation_end-total_evaluation_start)/60, "min")
        print("total time per doc: ", (total_evaluation_end-total_evaluation_start)/filtered_docs, "s", " = ", ((total_evaluation_end-total_evaluation_start)/filtered_docs)/60, "min")

        #time_sum = (
        #(split_end-split_start) + 
        #(base_end-base_start) + 
        #(initial_ft_end-initial_ft_start) +
        #(al_efficient_end-al_efficient_start) +
        #(al_random_end-al_random_start))

        #print(split_end-split_start)
        #print(base_end-base_start)
        #print(initial_ft_end-initial_ft_start)
        #print(al_efficient_end-al_efficient_start)
        #print(al_random_end-al_random_start)
        #print("total time elapsed for evaluation: ", time_sum, "s")




def get_other_labels():
    time_start = time.time()
    print(get_json(base_url+"/get_other_labels/"))
    print(get_json(base_url+"/get_other_labels/"))
    time.sleep(10)
    print(get_json(base_url+"/get_other_labels/"))

    print("time elapsed (per get labels (actual call)): " + str((time.time()-time_start-10)/2))

    #print(get_json(base_url+"/count_other_labels/"))

    
def test_lots_of_data():
    print(get_json(base_url+"/get_annotation_session_statistics/"))
    store_db_to_folders()
     


blocked_ids = {}

def add_id(id):
    global blocked_ids
    blocked_ids[id] = time.time()
    print("add id" +str(id))
    # remove ids that are no longer blocked

    print("time now " +str(time.time()))
    print("time stored +10" +str(blocked_ids[id] + 10))

    copy_blocked_ids = {}
    for id in blocked_ids:
        if time.time() < blocked_ids[id] + 4:
            copy_blocked_ids[id] = blocked_ids[id]
            print("add"+str(id))
        else:
            print("remove"+str(id))
        
    blocked_ids = copy_blocked_ids


def get_blocked_ids():
    list_of_blocked_ids = []
    
    for id in blocked_ids:
        # check time
        # if time now is less than time (id was saved + 30 mins)
        # include in blocked_ids 
        if time.time() < blocked_ids[id] + 10:
            list_of_blocked_ids.append(id)

    return list_of_blocked_ids


def t_30_min_list():
    ids = ["1","2", "3"]
    for id in ids:
        add_id(id)

    time.sleep(5)
    add_id("4")

    print("get blocked ids")
    print(get_blocked_ids())
    time.sleep(9)
    print(get_blocked_ids())

    
# multiple users click get next at the same
# time but should not receive the same id

list_of_docids = []
def test_race_condition_in_get_next(queue_type):

    print(post_json({}, base_url+"/init_next_most_important/"+queue_type))

    for i in range(1):
        result = json.loads(get_json(base_url+"/next_most_important/"+str(i)))
        print("annotator ",i)
        print(result)

    threads = []
    for annotator_id in range(4):
        t = threading.Thread(target = thread_action, args=(annotator_id,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join #?

    time.sleep(20) # sleep so that threads can still run before test is over
    repeated_entry_in_list(list_of_docids)
    print("total docs shown " + str(len(list_of_docids)))
    

def thread_action(annotator_id):
    for i in range(10):
        #print(str(annotator_id) + " tries to get next " + str(time.time()))
        result = json.loads(get_json(base_url+"/next_most_important/"+str(annotator_id)))
        if result != None and result != "{}" and result != {}:
            print(str(annotator_id) + " gets: " +str(result))

            document = json.loads(get_json(base_url+"/get_document/"+result["document_id"]))
            print("target_collection ",document["target_collection"])

            list_of_docids.append(result["document_id"])

        if result == "{}" or result == {} or result == "None":
            print(result)

def repeated_entry_in_list(list_name):

    for idx, viewed in enumerate(list_name):
        if [i for i, e in enumerate(list_name) if e == viewed] != [idx]:
            print("repeated: " + str(viewed))
        assert([i for i, e in enumerate(list_name) if e == viewed] == [idx])


def al_complete_show_partial():
    #print(post_json({}, base_url+"/init_next_most_important/al_complete"))
    #for i in range(5):
    #    result = json.loads(get_json(base_url+"/next_most_important/"+str(i)))
    #    print("annotator ",i)
    #    print(result)

    test_race_condition_in_get_next("al_complete")

    blocked_ids = json.loads(get_json(base_url+"/get_blocked_ids/"))
    #print(blocked_ids)
    print("blocked ids: " + str(len(blocked_ids)))

    #restart queue -> should not show in **blocked ids**
    print("restart")
    print(post_json({}, base_url+"/init_next_most_important/al_complete"))
    for i in range(5):
        doc = json.loads(get_json(base_url+"/next_most_important/"+str(1)))
        print(doc)
        if doc!= None and doc["document_id"] in blocked_ids:
            print("ERROR!!!! not blocked")



def label_statistics():
    result = json.loads(get_json(base_url+"/get_document_label_distribution/"))
    print(result)
    print("")

    result = json.loads(get_json(base_url+"/get_segment_label_distribution/"))
    print(result)
    print("")

    result = json.loads(get_json(base_url+"/get_document_type_distribution/"))
    print(result)
    print("")

    result = json.loads(get_json(base_url+"/get_segment_type_distribution/"))
    print(result)
    print("")

    result = json.loads(get_json(base_url+"/get_label_and_type_distribution/"))
    print(result)
    

def put_data_in_data_base_no_delete():
    global NO_CLEANUP_AFTER
    NO_CLEANUP_AFTER = True


def cursor_timeout():

    print(post_json({}, base_url+"/init_next_most_important/"+"al_complete"))

    result = json.loads(get_json(base_url+"/next_most_important/"+str(1)))
    print(result)
    print("wait")

    time.sleep(60*15) 
    # expected: cursor timeout -> nope

    result = json.loads(get_json(base_url+"/next_most_important/"+str(2)))
    print(result)

    time.sleep(60*15) 
    # expected: cursor timeout -> nope

    result = json.loads(get_json(base_url+"/next_most_important/"+str(2)))
    print(result)
    

t1 = [
    ("T1: add files to database", add_files_to_collection, ["priority", "priority"], {}),
    ("add files to database", add_files_to_collection, ["other", "other"], {}),
    ("initialize initial queue", initial_queue_init, [], {}),
    ("get statistics", get_statistics, [], {}),
    ("run initial queue", initial_queue_run_full, [], {})
]

t3 = [
    ("T3: copy files to complete", add_files_to_collection, ["complete", "complete"], {}),
    ("get statistics", get_statistics, [], {}),
    ("complete queue auto reset", complete_queue_reset, [], {})
]

t4 = [
    ("T4: copy files to partial", add_files_to_collection, ["partial", "partial",5], {}),
    ("copy files to complete", add_files_to_collection, ["complete", "complete",5], {}),
    ("get statistics", get_statistics, [], {}),
    ("run main", main_queue_run, [], {}),
    #("run main 15%", main_queue_run_until_15_percent_2nd_annotation, [], {})
]

t5 = [
    ("T5:  copy files to complete", add_files_to_collection, ["complete", "complete", 10], {}),
    ("copy files to total_annotation", add_files_to_collection, ["total_annotation", "total_annotation", 40], {}),
    ("get statistics", get_statistics, [], {}),
    ("run total queue", total_queue_run, [], {})
]

t6 = [
    ("T6: copy files to ml_out", add_files_to_collection, ["testing", "ml_out", 10], {}),
    ("run entropy queue", ml_queue_run, [], {})
]

t8 = [
    # compare speed of datagenerator in mongodb vs jsondatastore
    #("T8: test mongodb", test_mongodb, ["testing"], {}) # TODO stop jsondatastore from writing to test folder
    ]

t9 = [
    ("T9: add_files_to_collection", add_files_to_collection, ["testing", "priority"], {}),
    ("initialize queue", initial_queue_init, [], {}),
    ("test score based queue generation", basic_queue_generation, [], {})
    ]

t10 = [
    ("T10: add_files_to_collection", add_files_to_collection, ["testing", "ml_out"], {}),
    ("initialize queue", queue_init, ["al_entropy"], {}),
    ("test score based queue generation", score_based_queue_generation, [], {})
    ]

t11 = [
    # test queue
    ("T11: add_files_to_collection", add_files_to_collection, ["ml_out", "ml_out", 6], {}),
    ("test queue", test_queue_status, [], {})
    ]

t12 = [
    # test larger queue
    ("T12: add_files_with_status", add_files_to_different_collections, ["testing", 600], {}), # add to priority, other, partial
    ("test large queue", test_large_queue, [], {})
    ]

t13 = [
    ("T13: store data", add_files_to_different_collections, ["testing", 50], {}),
    ]

t14 = [
    ("T14: test generation of queue condition", test_queue_condition, [], {}),
    ]

    
t15 = [
    ("T15: save all files in a folder", save_to_folder, [], {}),
    ]

        
t16 = [
    ("T16: add_files_to_collection", add_files_to_collection, ["ml_out", "ml_out", 50], {}),
    ("test_scoring_functions", test_scoring_functions, [], {})
    ]

t17 = [
    ("T17: add_files_to_collection", add_many_docs, [5000,500, "ml_out", "ml_out"], {}),
    ("measure_runtime", measure_runtime, [], {})
]

t18 = [
    ("T18: add_files_to_collection", add_many_docs, [500000,800, "ml_out", "ml_out"], {}),
    ("measure_runtime", measure_runtime, [], {})
]

t19 = [
    ("T19: add files", add_files_to_collection, ["ml_out", "ml_out", 6], {}),
    ("test queue ml_out_diversity", test_ml_out_diversity, [], {})
]

t20 = [
    ("add files", add_files_to_collection, ["complete", "complete"], {}),
    ("T20: get human annotation", get_human_annotation_basic, [], {}),
]

t21 = [
    ("start_prediction", start_prediction, [], {}), 
]

t22 = [
    ("add files", add_files_to_collection, ["complete", "complete", 50], {}),
    ("T22: human_annotation", human_annotations, [], {}), 
]

t23 = [
    ("add files", add_files_to_collection, ["complete", "has_complete_human_annotation", 50], {}),
    ("T23: human_annotation", split_dataset, [], {}), 
]

t24 = [
    ("add files", add_files_to_collection, ["ml_out", "ml_out", 5], {}),
    ("add files", add_files_to_collection, ["complete", "has_complete_human_annotation", 5], {}),
    ("T24: test_store_db_to_folders", test_store_db_to_folders, [], {}),    
]

t24_large = [
    #("add_files_to_collection", add_many_docs, [25000,800, "ml_out", "ml_out"], {}),
    #("add_files_to_collection", add_many_docs, [25000,100, "complete", "complete"], {}),
    ("add_files_to_collection", add_many_docs, [250,800, "ml_out", "ml_out"], {}),
    ("add_files_to_collection", add_many_docs, [250,100, "complete", "complete"], {}),
    
    ("t24_large: test_store_db_to_folders", test_lots_of_data, [], {}),    
]


t25 = [
    #("add_files_to_collection", add_many_docs, [10000,800, "ml_out", "ml_out"], {}),
    #("add_files_to_collection", add_files_to_collection, ["ml_out", "ml_out", 200],  {}),

    ("add_files_to_collection", add_many_docs, [600,60, "partial", "has_partial_human_annotation"], {}),
    ("add_files_to_collection", add_many_docs, [400,40, "complete", "has_complete_human_annotation"], {}),

    #("add files", add_files_to_collection, ["partial", "has_partial_human_annotation", 60], {}),
    #("add files", add_files_to_collection, ["complete", "has_complete_human_annotation", 40], {}),

    ("T25: test evaluation", test_evaluation, [], {}), 
    #("store to folders", store_db_to_folders(), [], {}), 
]

t25_small = [

    ("add_files_to_collection", add_many_docs, [30,30, "partial", "has_partial_human_annotation"], {}),
    ("add_files_to_collection", add_many_docs, [20,20, "complete", "has_complete_human_annotation"], {}),
    #("add_files_to_collection", add_many_docs, [20,20, "ml_out_200916", "ml_out_200916"], {}),
    ("T25: test evaluation", test_evaluation, [], {}), 
]

t25_too_small = [

    ("add_files_to_collection", add_files_to_collection, ["complete", "has_complete_human_annotation", 1], {}),
    ("T25: test evaluation", test_evaluation_too_small, [], {}), 
]

t26 = [
    ("add files", add_files_to_collection, ["ml_out", "has_complete_human_annotation",20], {}),
    ("T26: test eval random", test_eval_random, [], {}),    
]

t27 = [
    ("add files", add_files_to_collection, ["ml_out", "ml_out", 10], {}),
    ("T27: test diversity queue", test_diversity_queue_in_db, [], {}),    
]


t28 = [
    ("add_files_to_collection", add_many_docs, [50000,500, "ml_out", "ml_out"], {}),
    ("T28: test diversity queue", test_diversity_queue, [], {}),    
]


t29 = [
    #("add files", add_files_to_collection, ["ml_out", "ml_out", 100], {}),
    ("add files", add_files_to_collection, ["ml_out", "ml_out", 20], {}),
    ("T29: test_diversity_queue_small", test_diversity_queue_small, [], {}),    
]

t30 = [
    ("add files", add_files_to_collection, ["annotation_version_test", "annotation_version_test"], {}),
    ("T30: test_annotation_version", test_annotation_version, [], {}),    
]

t31 = [
    #("add files", add_files_to_collection, ["annotation_version_test", "annotation_version_test"], {}),
    #("T31: test_annotation_version", test_extend_annotation_vectors, [], {}),    
    #("store to folder", store_db_to_folders, [], {}),    
]


t32 = [
    ("add files", add_many_docs, [80,80, "ml_out", "ml_out"], {}),
    ("T32: test_annotation_session", test_annotation_session, [], {}),    
]

t33 = [
    ("T33: test_filtered_finetuned_annotation_types", test_filtered_finetuned_annotation_types, [], {}),    
]

t34 = [
    ("add files", add_files_to_collection, ["ml_out", "ml_out", 10], {}),
    ("T34: test_filtered_finetuned_annotation_types", run_partial_queue, [], {}),    
    ("store_db_to_folders", store_db_to_folders, [], {}),    
]

t35 = [
    ("add files", add_files_to_collection, ["annotation_version_test", "annotation_version_test"], {}),
    ("try_to_get_invalid_id", try_to_get_invalid_id, [], {}),
]

t36 = [
    ("add files", call_rest_calls_in_document_annotator, [], {}),
]

t37 = [
    ("test_random", test_random, [], {}),
]

empty_log_files = [
    ("empty_log_files",  empty_log, [], {}),
]

t38 = [
    ("next_most_important before queue init",  next_most_important_before_queue_init, [], {}),
]

t39 = [
    ("add files", add_files_to_collection, ["ml_out", "ml_out", 10], {}),
    ("add files", add_files_to_collection, ["complete", "complete", 20], {}),
    ("get number of docs",  get_number_of_docs, [], {}),
]

t40 = [
    ("add files", add_files_to_collection, ["has_complete_human_annotation", "has_complete_human_annotation", 10], {}),
    ("add files", add_files_to_collection, ["has_partial_human_annotation", "has_partial_human_annotation", 10], {}),
    ("add files", add_files_to_collection, ["ml_out_200916", "ml_out", 10, None], {}),
    ("add files", add_files_to_collection, ["partial", "partial", 10], {}),
    ("add files", add_files_to_collection, ["other", "other", 5], {}),
    ("test al_view_finetuned",  test_al_view_finetuned, [], {}),
]

t41 = [
    ("add files", add_files_to_collection, ["has_complete_human_annotation", "has_complete_human_annotation"], {}),
    ("test accuracy",  test_accuracy, [], {}),
]

t42 = [
    ("add files", add_files_to_collection, ["has_complete_human_annotation", "has_complete_human_annotation"], {}),
    ("test start_finetuned_predictions",  test_start_finetuned_predictions, [], {}),
]

t43 = [
    ("add files", add_files_to_collection, ["has_complete_human_annotation", "has_complete_human_annotation",2], {}),
    ("test skipping ids",  test_skipping_ids, [], {}),
]


t44 = [
    ("read times from log",  read_times_from_log, [], {}),
]

t45  = [
    ("add files", add_files_to_collection, ["complete", "complete"], {}),
    ("get HA, missing file",  get_human_annotation_missing_file, [], {}),
]

t46  = [
    ("add files", add_many_docs, [30, 500, "has_complete_human_annotation", "has_complete_human_annotation"], {}),
    ("add files", add_many_docs, [30, 500, "has_partial_human_annotation", "has_partial_human_annotation"], {}),
    ("add files", add_many_docs, [50, 500, "ml_out", "ml_out"], {}),
    ("get other labels",  get_other_labels, [], {}),
]

t_30_min_list = [
    ("t_30_min_list",  t_30_min_list, [], {}),

]

t_no_repeat_in_partial = [
    ("add files", add_many_docs, [1000, 100, "ml_out", "ml_out"], {}),
    ("no_repeat_in_partial",  no_repeat_in_partial, [], {}),    
]

t_test_race_condition_in_get_next = [
    ("add files", add_many_docs, [100, 100, "ml_out", "ml_out"], {}),
    ("t_test_race_condition_in_get_next",  test_race_condition_in_get_next, ["al_partial"], {}),    
]

t_al_complete_show_partial = [
    ("add files", add_many_docs, [100, 100, "has_partial_human_annotation-WS-2", "has_partial_human_annotation"], {}),
    ("al_complete_show_partial",  al_complete_show_partial, [], {}),    
]

t_label_statistics = [
    ("add files", add_many_docs, [100, 100, "has_partial_human_annotation-WS-2", "has_partial_human_annotation"], {}),
    ("add files", add_many_docs, [100, 100, "has_complete_human_annotation", "has_complete_human_annotation"], {}),
    ("label_statistics",  label_statistics, [], {}),    

]


test_cursor_timeout = [
    ("add files", add_many_docs, [10, 10, "has_partial_human_annotation-WS-2", "has_partial_human_annotation"], {}),
    ("test_cursor_timeout",  cursor_timeout, [], {}),    

]

put_data_in_data_base_no_delete= [
    ("add files", add_many_docs, [30, 30, "has_partial_human_annotation", "has_partial_human_annotation"], {}),
    ("add files", add_many_docs, [30, 30, "has_complete_human_annotation", "has_complete_human_annotation"], {}),
    ("put_data_in_data_base_no_delete",  put_data_in_data_base_no_delete, [], {}),    
]

reformat_jsons = [
    ("add files", add_files_to_collection, ["has_complete_human_annotation", "has_complete_human_annotation"], {}),
    ("reformat_jsons",  reformat_jsons, [], {}),    
    #("store_db_to_folders",  store_db_to_folders, [], {}),    
]

tests = [
    #t1, # OK (added priority/other in test folder, test that initial importance is either 0 or 1)
    #t3, # OK  
    #t4, # OK  
    #t5, # OK  # need total_annotation folder
    #t6, # OK # needs folder testing 
    #t8, # OK # needs folder testing
    #t9, # OK
    #t10, # OK 
    #t11, # OK --> needs ml test data (in ml_out folder)
    #t12, # OK (600 docs)  
    #t13, # OK store 50 files 
    #t14, # OK
    #t15,
    #t16,
    ##t17, takes very long
    ##t18, takes very long
    #t19,
    #t20,
    #t22,
    #t23,
    #t25, # test evaluation
    #t25_too_small
    #t26,
    #t27,
    #t28,# test diversity queue
    #t29,
    #t30,
    #t31 #DEBUG_extend_annotation_vectors in REST_AL must be True
    #t33
    #t34,
    #t35
    #t36
    #t37
    #t38
    #t39
    #t45 # get human annotation, missing file
    #t46

    #t21, #start prediction
    #t24, # test store to folders
    #t24_large
    #t25_small, # test evaluation
    #t32, # test_annotation_session, al_view_queue, switch partial complete
    #t40, # test finetune view queue
    #t41,
    #t42, # finetuned predictions
    #t43, # skip ids
    #t44, # read times from log
    #empty_log_files

    #t_30_min_list,
    #t_no_repeat_in_partial
    #t_test_race_condition_in_get_next,
    #t_al_complete_show_partial
    #test_cursor_timeout
    #t_label_statistics

    #put_data_in_data_base_no_delete
    reformat_jsons
]