import time
import pymongo 
import os
import json
import traceback
import threading
from glob import glob

from efficient_annotation.common import Document, JSONSerializable
from efficient_annotation.logging import Logger
from efficient_annotation.common import load_config_file


class MongoDB:
    # Handles file storage and access

    def __init__(self, annotation_types, collections, database_url):  
        
        print('start database')
        is_docker = os.environ.get('IS_DOCKER', False)

        # start up databasee
        if is_docker:
            print('database is running in a Docker container')
            # document-database is the name of the database service -> see docker-compose.yml
            # ensures that connection is correct in docker container
            self.client = pymongo.MongoClient(database_url)
        else: 
            print('database is not running in a Docker container')
            self.client = pymongo.MongoClient("mongodb://localhost:27017/")

        self.database = self.client["database"]
        self.document_collection = self.database["documents"]

        self.annotation_types = annotation_types
        self.collections = collections
        self.queue_manager = None
        self.VERBOSE = True    
        self.logger = Logger()
        self.lock = threading.Lock()
        self.list_of_blocked_ids = []
    
    # register the queue_manager
    def register_queue_manager(self, queue_manager):
        self.queue_manager = queue_manager

    # return true if document with document_id is in collection
    def is_in_collection(self, document_id, collection):
        query = { "_id": document_id}
        doc = self.document_collection.find_one({query},{ "target_collection": 1})
        return doc["target_collection"] == collection

        
    # save json in database
    def save_to_collection(self, document, collection = None):
        
        if  isinstance(document, Document):
            document = Document.serialize(document)

        if "target_collection" not in document and collection == None:
            document["target_collection"] = "other"
            print("WARNING: document json file did not contain target_collection. set target_collection to other", flush = True) 

        if collection != None:
            document["target_collection"] = collection 

        # add annotators
        document["annotated_by"]["annotators_partial"] = []
        document["annotated_by"]["annotators_complete"] = []
        
        flag_old_format = False

        for category in ["document_contains", "document_type", "document_label"]:
            if document["annotated_by"][category] != []:

                # handle old json-format where annotators are in list: ["name1", "name2", ...]
                # instead of: [{"name": "name1", "timestamp": "<timestamp>"}, {"name": "name2", "timestamp": "<timestamp>"}, ]

                # ------- if old format: convert to new format  --------------------------- 
                if isinstance(document["annotated_by"][category][0], str):
                    flag_old_format = True
                    annotator_list_new_format = []

                if flag_old_format == True:
                    for annotator in document["annotated_by"][category]:
                        annotator_list_new_format.append({"name": annotator, "timestamp": ""})
                    document["annotated_by"][category] = annotator_list_new_format
                # -------------------------------------------------------------------------

                for annotator in document["annotated_by"][category]:
                    if not annotator["name"] in document["annotated_by"]["annotators_partial"]:
                        document["annotated_by"]["annotators_partial"].append(annotator["name"])


        for category in ["is_occluded", "segment_boundary", "segment_label", "segment_type"]:
            if document["annotated_by"][category] != []:

                # ------- if old format: convert to new format  --------------------------- 
                if isinstance(document["annotated_by"][category][0], str):
                    flag_old_format = True
                    annotator_list_new_format = []

                if flag_old_format == True:
                    for annotator in document["annotated_by"][category]:
                        annotator_list_new_format.append({"name": annotator, "timestamp": ""})
                    document["annotated_by"][category] = annotator_list_new_format
                # -------------------------------------------------------------------------

                for annotator in document["annotated_by"][category]:
                    if not annotator["name"] in document["annotated_by"]["annotators_complete"]:
                        document["annotated_by"]["annotators_complete"].append(annotator["name"])


        if flag_old_format == True:
            print("WARNING: document json file annotated_by in old format. annotators should have name and timestamp")

        #print("annotators_partial")
        #print(document["annotated_by"]["annotators_partial"])

        #print("annotators_complete")
        #print(document["annotated_by"]["annotators_complete"])

        # TODO: we add field _id here which is the same at document_id
        # but this redundancy might be confusing
        document["_id"] = document["document_id"]
        
        # inserts document if it does not exists or updates existing document with same id
        query = { "_id": document["_id"] }

        new_entry = { "$set": document }
        self.document_collection.update_one(query, new_entry, upsert=True)

        # write json file to folder bound on local host
        #is_docker = os.environ.get('IS_DOCKER', False)

        # TODO include again to write to data folder
        #if (not is_docker):
        #    path = os.path.join(os.getcwd(), "\mongodb\\", str(document["document_id"])+".json") #, self.collections[document["target_collection"]]['path']
        
        # write_to_files = False
        # if (write_to_files):
        #     if (is_docker):
        #         path = os.path.join(os.getcwd(), '../data', str(document["document_id"])+".json")
        #     with open(path, 'w') as out_file:
        #         json.dump(document, out_file)

    def save_all_documents_in_folder(self, main_folder):

        print("current folder: " + os.getcwd(), flush = True)
        print("inside current folder: " + str(os.listdir(os.getcwd())), flush = True)

        main_folder = main_folder.replace("'","")

        #copy json files from subfolders (collection folders) from main_folder
        sub_folders = glob(main_folder + "/*/")

        try:
            for subfolder in sub_folders:
                #subfolder = os.path.join(main_folder, subfolder)
                print('store all files inside ' + subfolder + ' in mongodb', flush = True)

                fnames = os.listdir(subfolder)
                fnames = set([fname for fname in fnames if fname.endswith(".json")])
                for fname in fnames:
                    path = os.path.join(subfolder, fname)
                    if os.path.isfile(path) == True:
                        print('save: ' + path)
                        with open(path, "r") as fp:
                            document = json.load(fp)
                            print("collection = " + str(subfolder.replace(main_folder,'').replace('/','')))
                            self.save_to_collection(document, subfolder.replace(main_folder,'').replace('/','') )
        except Exception as e:
            traceback.print_exc()

        # copy json files that are directly in main_folder
        try:
            print('store all files inside top level of folder ' + main_folder + ' in mongodb', flush = True)
            fnames = os.listdir(main_folder)
            fnames = set([fname for fname in fnames if fname.endswith(".json")])
            for fname in fnames:
                path = os.path.join(main_folder, fname)
                if os.path.isfile(path) == True:
                    print('save: ' + path, flush = True)
                    with open(path, "r") as fp:
                        document = json.load(fp)
                        self.save_to_collection(document)
        except Exception as e:
            #traceback.print_exc()
            print("no json files in top level of " + main_folder)


    # computes scores for documents in collection using scoring_function
    # and returns documents in collection sorted by this score 
    def add_scores(self, scoring_function, weights = None):

        # compute score with scoring_function for all documents in target_collection = collection

        '''
        # use this if we want to add different importance scores per document
        score_name = scoring_function.__name__
        for doc in self.document_collection.find(query):
            query = { "_id": doc.id}
            new_entry = { "$set": doc }
            self.document_collection.update_one(query, new_entry, upsert=True)
            self.document_collection.update_one({"_id": doc["_id"]}, {"$set": {score_name: scoring_function(doc)}}, upsert=True)
        
        # sort documents by score
            queue = self.document_collection.find(query).sort(score_name, -1)
        '''

        # TODO?: for speed make scoring_function for json

        # update importance_score
        for doc in self.document_collection.find():
            document_obj = Document.deserialize(doc) 
            query = { "_id": doc["_id"] }
            if (weights != None):
                new_values = { "$set": { "importance_score": scoring_function(document_obj, weights) } }
            else:
                new_values = { "$set": { "importance_score": scoring_function(document_obj)} }
            self.document_collection.update_one(query, new_values)

        # return document_id and importance_score of documents sorted by importance_score
        #queue = self.document_collection.find({}, {"_id": 1, "importance_score": 1}).sort("importance_score", -1)
        #return queue


    # add a field document_label_predicted_by_base_model
    # to quickly retrieve documents with a specific predicted document label
    #TODO test
    # use with collection ml_out
    def add_document_label_predicted_by_base_model(self, collection):
        collection_query = {"target_collection":collection}

        for doc in self.document_collection.find(collection_query):
            for document_label_annotation in doc["document_label"]["annotations"]:
                if document_label_annotation["annotator_id"] == "segment_multimodal_model":
                    # get index of maximum class label score
                    vec = document_label_annotation["annotation_vector"]
                    maximum = max(vec)

                    # get the index of the maximum value in the annotation vector
                    index = [i for i, j in enumerate(vec) if j == maximum] 
                    # check that there is exactly one maximum and that index is desired_index
                    # i.e. the predicted class of this document is (potentially) the class we want
                    
                    # TODO could set a threshold for minimum value of maximum
                    # e.g. 0.5
                    query = { "_id": doc["_id"] }
                    if len(index) == 1 and maximum > 0:
                        new_values = { "$set": { "document_label_predicted_by_base_model": index[0]} }
                        self.document_collection.update_one(query, new_values)
                    else:
                        new_values = { "$set": { "document_label_predicted_by_base_model": -1} }
                        self.document_collection.update_one(query, new_values)


    # find Document by id
    def load_from_datastore(self, document_id):
        query = { "_id": document_id}
        document_json = self.document_collection.find_one(query)
        return Document.deserialize(document_json)


    def get_document_collection(self, document_id):
        include_values = {"_id": 1, "target_collection": 1}
        return self.document_collection.find_one({ "_id": document_id}, include_values)

    def load_from_datastore_as_json(self, document_id):
        return self.document_collection.find_one({ "_id": document_id})


    # returns document_id, importance_score, initial_importance and annotated_by
    # sorted in descending order of importance_score
    def get_document_importance_from_datastore(self, collection):
        query = { "target_collection": collection}
        include_values = { "_id": 1, "importance_score": 1, "initial_importance": 1, "annotated_by": 1 }
        return self.document_collection.find(query,include_values).aggregate([{ "$match": query }, {"$sort":{"importance_score":-1}}], allowDiskUse=True) #.sort("importance_score", -1)


    # check if id of annotator is among annotator_id in json
    # used to create queue
    # num_annotators number of annotators that have seen this document before annotator
    # is used to get first documents seen by one annotator than two annotators
    def get_document_importance_from_datastore_not_annotated_by(self, num_annotators, annotator, collection_order = None):
        #print("mongodb: get docs not annotated by annotator with id: " + str(get_annotator()), flush = True)

        # note: we could either say someone can see the document if they
        # are in none of the relevant annotated_by categories
        # => will not see a document twice during an annotation phase (partial, complete...)
        # pro: document not shown again if it was left unfinished on purpose
        # or not in all of them
        # => can see a document again during an annotation phase and add missing parts
        # pro: no document "dissapears" if it was left unfinished

        # ==> i will go with: 
        # show if annotator in none of the relevant categories
        # and number of annotators should be over all relevant categories

        # partial -> relevant: annotated_by.document_contains / document_type / document_label
        # complete -> relevant: annotated_by.is_occluded / segment_boundary / segment_label / segment_type
        
        # only show the document to another annotator if it is in the "complete phase"
        # show document for partial annotation only to one annotator
        #query = {
        #    # annotator has not seen document during complete phase
        #    "$and": [ 
        #    {"target_collection": { "$in": ["complete", "2nd_annotation", "3rd_annotation", "total_annotation", "total_priority", "ml_out", "al_entropy", "al_random"]}},   
        #    {"annotated_by.annotators_complete": { "$ne": annotator }}, 
        #    {"annotated_by.annotators_complete": {"$size" : num_annotators}}
        #    ]
        #}
        
        query = {
            "$or": [
                # annotator has not seen document during partial phase (do not show priority/other (=docs with no annotation at all) again)
                {
                "$and": [ 
                {"target_collection": { "$in": ["partial", "ml_out", "al_entropy", "al_random"]}}, 
                {"annotated_by.annotators_partial": { "$ne": annotator }}, 
                {"annotated_by.annotators_partial": {"$size" : num_annotators}}
                ]},
                # annotator has not seen document during complete phase
                {
                "$and": [ 
                {"target_collection": { "$in": ["complete", "2nd_annotation", "3rd_annotation", "total_annotation", "total_priority", "ml_out", "al_entropy", "al_random"]}},   
                {"annotated_by.annotators_complete": { "$ne": annotator }}, 
                {"annotated_by.annotators_complete": {"$size" : num_annotators}}
                ]},

                #{
                #"$and": [ 
                #{"target_collection": { "$in": ["partial"]}},   
                #{"annotated_by.annotators_complete": { "$ne": annotator }}, 
                #{"annotated_by.annotators_partial": {"$size" : num_annotators}}
                #]},
            ]
        }

        if (collection_order == None):
            collection_order = ["priority", "other", "partial", "complete"]

        # sort by target_collection, then by importance_score
        # reference: https://koenaerts.ca/mongodb-aggregation-custom-sort/
        return self.document_collection.aggregate([
            { "$match": query }, 
            # sort by queue status
            {"$project":{
                "_id":True,
                "sortField":
                self.generate_order_by_collection(collection_order),
                "importance_score" : True,
                "target_collection": True
            }},

            {"$sort":{"sortField":1, "importance_score":-1}}

            ], 
        allowDiskUse=True)

        #return self.document_collection.find(query, include_values).sort( [("status", 1), ("importance_score", -1)]) 

    # used to create queue
    def get_documents_without_human_annotation(self, collection_order = None):
        print("mongodb: get docs without human annotation: ", flush = True)

        # TODO handle ml_out, al_entropy, al_random (maybe put as a flag that this is machine learning output)
        # for now we put it in both partial and complete collections
        # find documents where annotators is empty list, None or non-existant
        # query = {"annotated_by.annotators": { "$in": [[], None]}}  

        if (collection_order == None):
            collection_order = ["priority", "other", "partial", "complete", "total_priority", "total_annotation"]

        query = {
            "$or": [
                # no annotator has seen document during partial phase
                {
                "$and": [ 
                {"target_collection": { "$in": ["priority", "other", "partial", "ml_out", "al_entropy", "al_random"]}},
                {"annotated_by.annotators_partial": { "$in": [[], None]}}, 
                ]},
                # no annotator has seen document during complete phase
                {
                "$and": [ 
                {"target_collection": { "$in": ["complete", "2nd_annotation", "3rd_annotation", "total_annotation", "total_priority", "ml_out", "al_entropy", "al_random"]}},   
                {"annotated_by.annotators_complete": { "$in": [[], None]}}, 
                ]},
            	]
        }

        # sort by target_collection, then by importance_score
        return self.document_collection.aggregate([
            { "$match": query }, 
            {"$project":{
                "_id":True,
                "sortField":
                self.generate_order_by_collection(collection_order),
                "importance_score" : True,
                "target_collection": True
            }},
            {"$sort":{"sortField":1, "importance_score":-1}},
            #{"$limit": 10}, # does not improve speed
        ], 
        allowDiskUse=True)      
        #print("TEST WITHOUT SORT", flush= True)
        

    def get_documents_without_complete_human_annotation(self):
        print("mongodb: get docs without complete human annotation: ", flush = True)

        # TODO handle ml_out, al_entropy, al_random (maybe put as a flag that this is machine learning output)
        # for now we put it in both partial and complete collections
        # find documents where annotators is empty list, None or non-existant
        #query = {"annotated_by.annotators": { "$in": [[], None]}}  

        query = {
            "$and": [ 
            {"target_collection": "partial"},   
            {"annotated_by.annotators_complete": { "$in": [[], None]}}, 
            {"annotated_by.annotators_partial": {"$size" : 1}}
            ]
        }

        include_values = { "_id": 1, "importance_score": 1, "target_collection": 1}
        return self.document_collection.find(query, include_values).aggregate([{ "$match": query }, {"$sort":{"importance_score":-1}}], allowDiskUse=True) #.sort("importance_score", -1)


    # creates conditions to order documents by target_collection
    # e.g. collection_order = ["priority", "other", "partial"]
    # in this example the documents will be put in order "priority" -> "other" -> "partial" -> the rest
    #{"$cond":[{"$eq":["$target_collection", collection_order[0]]}, 1,
    #    {"$cond":[{"$eq":["$target_collection", collection_order[1]]}, 2,  
    #        {"$cond":[{"$eq":["$target_collection", collection_order[2]]}, 3, 4  
    #        ]} 
    #    ]} 
    #]}
    def generate_order_by_collection(self, collection_order):
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

        return  complete_wrapper


    # return all documents in collection
    def get_all_documents_in_collection(self, collection):
        print("return all documents in target_collection: " + collection)
        query = { "target_collection": collection}
        return self.document_collection.find(query)

    def get_documents_by_query(self, query): 
        return self.document_collection.find(query)


    def get_all_document_ids_in_collection(self, collection):
        print("return all documents in target_collection: " + collection)
        query = { "target_collection": collection}
        return self.document_collection.find(query,  {"_id": 0, "document_id": 1})


    # return all documents in datastore
    def get_all_documents(self):
        return self.document_collection.find()


    # find all documents in collection
    # return sort by importance_score if sort = True
    def document_generator(self, collection, sort = True):
        query = { "target_collection": collection}
        if sort:
            for document_json in self.document_collection.aggregate([{ "$match": query }, {"$sort":{"importance_score":-1}}], allowDiskUse=True): # find(query).sort("importance_score", -1)
                document = Document.deserialize(document_json)
                yield document
        else:
            for document_json in self.document_collection.find(query):
                document = Document.deserialize(document_json)
                yield document

    def document_generator_from_query(self, query, sort = True):
        if sort:
            for document_json in self.document_collection.aggregate([{ "$match": query }, {"$sort":{"importance_score":-1}}], allowDiskUse=True):  #find(query).sort("importance_score", -1)
                document = Document.deserialize(document_json)
                yield document
        else:
            for document_json in self.document_collection.find(query, no_cursor_timeout=True):
                document = Document.deserialize(document_json)
                yield document


    def document_generator_from_multiple_collections(self, collections, sort = True):
        query = { "target_collection": { "$in": collections } }
        if sort:
            for document_json in self.document_collection.aggregate([{ "$match": query }, {"$sort":{"importance_score":-1}}], allowDiskUse=True): # find(query).sort("importance_score", -1):
                document = Document.deserialize(document_json)
                yield document
        else:
            for document_json in self.document_collection.find(query):
                document = Document.deserialize(document_json)
                yield document

    def get_num_docs_per_target_collection(self):
        result = []
        for target_collection in self.document_collection.distinct("target_collection"):
            num = self.get_number_of_documents(target_collection)
            result.append({"target_collection": target_collection, "num": num})

        return result 

    # change all target_collection of all documents with 
    # old_target_collection (e.g. "viewed") to new_target_collection (e.g. "ml_out")
    def set_target_collection_old_to_new(self, old_target_collection, new_target_collection):
        query = { "target_collection": old_target_collection}
        for document in self.document_collection.find(query):

            query = { "_id": document["_id"] }
            document["target_collection"] = new_target_collection
            new_entry = { "$set": document }
            self.document_collection.update_one(query, new_entry)



    # *** blocked ids *** 
    # dict id-time
    # if doc not None: add id/time to list 
    # get list of "bad ids": in list and now < time+(30*60)  
    # query: id cannot be in list

    blocked_ids = {}
    block_time_secs = 60*60

    def add_id(self, id):
        self.blocked_ids[id] = time.time()

        # remove ids that are no longer blocked
        copy_blocked_ids = {}

        for id in self.blocked_ids:
            if time.time() < self.blocked_ids[id] + self.block_time_secs:
                copy_blocked_ids[id] = self.blocked_ids[id]

        self.blocked_ids = copy_blocked_ids

        for id in self.blocked_ids:
            # check time
            # if time now is less than time (id was saved + 30 mins)
            # include in blocked_ids 
            if time.time() < (self.blocked_ids[id] + self.block_time_secs):
                if id not in self.list_of_blocked_ids: 
                    self.list_of_blocked_ids.append(id)

    def get_blocked_ids(self):
        #list_of_blocked_ids = []
        #for id in self.blocked_ids:
        #    # check time
        #    # if time now is less than time (id was saved + 30 mins)
        #    # include in blocked_ids 
        #    if time.time() < (self.blocked_ids[id] + self.block_time_secs):
        #        list_of_blocked_ids.append(id)

        return self.list_of_blocked_ids


    # updated version that should be faster
    def get_document_with_predicted_document_label_v2(self, collection, desired_label):
        
        # to prevent race conditions when multiple users 
        # try to get next document at the same time:
        # lock retrieving document and adding it to list of blocked documents
        with self.lock:
            # find the segment_multimodal_model annotations -> and check that the argmax has index desired_label
            query = {
                "target_collection": collection, 
                "document_label_predicted_by_base_model": desired_label,
                "document_id" : { "$nin": self.get_blocked_ids() } 
                } 
            
            include_values = { "_id": 1, "document_id": 1, "importance_score": 1, "target_collection": 1, "document_label": 1, "document_label_predicted_by_base_model":1}
            document_json = self.document_collection.find_one(query, include_values)
            #if document_json != None:
            #    self.logger.log("return document with document label: " + str(document_json["document_label_predicted_by_base_model"]))

            # add id to blocked ids if doc is not None
            if document_json != None:
                self.add_id(document_json["document_id"])


        return document_json # return None if there is no document that is predicted to have the desired class 


    # for diversity queue
    # always pick a document from a different document label class
    # i.e. find a document that has maximum value in annotation vector at index desired_label
    def get_document_with_predicted_document_label(self, collection, desired_label):

        # find the segment_multimodal_model annotations -> and check that the argmax has index desired_label
        query = {"target_collection": collection} 
        
        include_values = { "_id": 1, "document_id": 1, "importance_score": 1, "target_collection": 1, "document_label": 1}
        for document_json in self.document_collection.find(query, include_values):
            for document_label_annotation in document_json["document_label"]["annotations"]:
                if document_label_annotation["annotator_id"] == "segment_multimodal_model":
                    # get index of maximum class label score
                    vec = document_label_annotation["annotation_vector"]
                    maximum = max(vec)

                    # get the index of the maximum value in the annotation vector
                    index = [i for i, j in enumerate(vec) if j == maximum] 
                    # check that there is exactly one maximum and that index is desired_index
                    # i.e. the predicted class of this document is (potentially) the class we want
                    
                    # TODO could set a threshold for minimum value of maximum
                    # e.g. 0.5
                    if len(index) == 1 and index[0] == desired_label and maximum > 0:
                        return document_json 

        return None # return None if there is no document that is predicted to have the desired class


    #return list of filenames in collection
    def document_file_list(self, collection):
        #filelist = os.listdir(self.collections[collection]['path'])
        #filelist = [fname for fname in filelist if fname.endswith(".json")]
        #return filelist

        query = { "target_collection": collection}
        return self.document_collection.find(query,  {"_id": 1})


    def get_number_of_documents(self, collection = None):
        if collection == None:
            return self.document_collection.count_documents({})
        else:
            query = { "target_collection": collection}
            return self.document_collection.count_documents(query)



    # used for testing: only adds annotator_id to the list of annotators
    # e.g.: "annotated_by" : ["annotators": [{"annotator_id":annotator_0}, {"annotator_id":annotator_1}], ... ]
    #def add_annotation(self, document_id, annotator_id):
        # document = self.load_from_datastore_as_json(document_id)

        # annotated_by = document["annotated_by"]

        # try:     
        #     print("field annotators exits -> append new annotator")
        #     annotated_by["annotators"].append({"annotator_id":annotator_id})
        # except KeyError:
        #     print("field annotators does not exits")
        #     annotated_by["annotators"] = [{"annotator_id":annotator_id}]

        # query = { "_id": document_id }
        # new_entry = { "$set": document }
        # self.document_collection.update_one(query, new_entry, upsert=True)


    def empty_db(self):
        print("############################################################")
        print("WARNING!! FOR TEST PURPOSES ONLY -- DROP DATABASE COLLECTION")
        print("############################################################")
        self.document_collection.drop()        
        #self.document_collection = self.database["documents"]

    def update_timestamps(self):
        print("update_timestamps not implemented in mongodb")


    def save_all_in_db_to_folders(self):

        self.logger.log("mongodb: save_all_in_db_to_folders")

        # save to app/data -> bound to local folder docker/monogdb 
        # => will be app/data and docker/monogdb 

        is_docker = os.environ.get('IS_DOCKER', False)

        if (is_docker):
            data_path = self.get_data_path() # os.path.join(os.getcwd(), '../data/')

            # get all documents
            for doc in self.get_all_documents():

                collection_path = os.path.join(data_path,doc["target_collection"])
                
                # create subfolders
                if not os.path.exists(collection_path):
                    os.mkdir(collection_path)

                # store all documents in collection
                doc_path = os.path.join(collection_path,str(doc["document_id"])+".json")

                with open(doc_path, 'w') as out_file:
                    json.dump(doc, out_file)
        else:
            print("###### WARNING: you are trying to save db to folders. but code is not running in docker container ######")


    # remove document-files that are not in the correct collection folders
    def clean_collection_folders(self):

        is_docker = os.environ.get('IS_DOCKER', False)

        if (is_docker):
            data_path = self.get_data_path()
                    
            # get all folders
            data_dirs = os.listdir(data_path)

            for data_dir in data_dirs:
                data_dir = os.path.join(data_path, data_dir)
                if os.path.isdir(data_dir):

                    for fname in os.listdir(data_dir):
                        file_path = os.path.join(data_dir, fname)
                        
                        doc_id = fname2document_id(fname)
                        doc = self.get_document_collection(doc_id)

                        #remove doc if: 
                        # doc is in db 
                        # and target collection of doc in db is not same as foldername

                        if (os.path.isfile(file_path) == True
                            and doc != None 
                            and doc["target_collection"] != data_dir):
                        
                            os.remove(file_path)

        else:
            print("###### WARNING: you are trying to clean collection folders. but code is not running in docker container ######")


    def get_data_path(self):
        config = load_config_file()
        return config["data"]["path"] 


def fname2document_id(fname):
    return fname[:len(fname)-len(".json")]