import json
import pickle
import os
import os.path
from os import walk
import errno
import time
from efficient_annotation.common import get_annotation_completion, JSONSerializable
from efficient_annotation.common import Document

class JSONDatastore():
    # Handles file storage and access
    
    def __init__(self, annotation_types, collections):
        self.annotation_types = annotation_types
        self.collections = collections
        self.queue_manager = None
        for collection in self.collections:
            try:
                os.makedirs(self.collections[collection]['path'])
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        
    
    def load_score_file(self, collection):
        # load score file or create new
        path = os.path.join(self.collections[collection]['path'], 'scores')
        if os.path.isfile(path):
            with open(path, 'rb') as fp:
                document_scores = pickle.load(fp)
        else:
            document_scores = {}
        return document_scores
    
    def initialize_scoring(self):
        # initialize the scoring for each collection
        # requires a registered queue_manager
        for collection in self.collections:
            score_based_queues = self.queue_manager.get_score_based_queues(collection)
            # skip collections without score_based_queues
            if len(score_based_queues) == 0:
                continue
            # load score file or create new
            document_scores = self.load_score_file(collection)
            # load filenames
            filelist = self.document_file_list(collection)
            
            # calculate score for each queue for each missing file name
            # skip if there is already an entry for a document (TODO: handle new queues/scores)
            for fname in filelist:
                document_id = fname2document_id(fname)
                if document_id in document_scores:
                    continue
                # load document
                document = self.load_from_datastore(document_id)
                # calculate score for each queue
                queue_scores = {}
                for score_based_queue in score_based_queues:
                    score = score_based_queue.score_calculator.scoring_function(document)
                    queue_scores[score_based_queue.name] = score
                # set the scores
                document_scores[document_id] = queue_scores
            
            # delete all scores of missing files
            document_ids = set([fname2document_id(fname) for fname in filelist])
            keys = list(document_scores.keys())
            for document_id in keys:
                if document_id in document_scores and document_id not in document_ids:
                    del document_scores[document_id]
            
            # pickle score file
            path = os.path.join(self.collections[collection]['path'], 'scores')
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
            with open(path, 'wb') as fp:
                pickle.dump(document_scores, fp, protocol=pickle.DEFAULT_PROTOCOL)
    
    def register_queue_manager(self, queue_manager):
        # register the queue_manager
        self.queue_manager = queue_manager

    def is_in_collection(self, document_id, collection):
        path = os.path.join(self.collections[collection]['path'], document_id+'.json')
        return os.path.isfile(path)

    def save_to_collection(self, document, collection=None):
        VERBOSE = False

        # garantee that document is deleted from old datastore if 
        # annotation has reached a certain stage
        if "previous" in self.collections[collection]:
            to_remove = self.collections[collection]["previous"]
            for c in to_remove:
                self.remove_from_collection(document.document_id, c)
        
        # if there are score_based_queues for this collection
        score_based_queues = self.queue_manager.get_score_based_queues(collection)
        
        if VERBOSE: print('len(score_based_queues): '); print(len(score_based_queues))
        
        if len(score_based_queues) > 0:
            # update score
            # load score file
            document_scores = self.load_score_file(collection)
            
            # calculate score for each queue
            queue_scores = {}
            for score_based_queue in score_based_queues:
                score = score_based_queue.score_calculator.scoring_function(document)
                if VERBOSE: print('score_based_queue = '); print(score_based_queue)
                if VERBOSE: print('score = ' + score)
                queue_scores[score_based_queue.name] = score
            
            # set the scores
            document_scores[document.document_id] = queue_scores
            
            # add score to document
            # skip for now
            
            # pickle score file
            path = os.path.join(self.collections[collection]['path'], 'scores')
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
            with open(path, 'wb') as fp:
                pickle.dump(document_scores, fp, protocol=pickle.DEFAULT_PROTOCOL)
        
        # save file to collection
        path = os.path.join(self.collections[collection]['path'], document.document_id+'.json')
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(JSONSerializable.serialize(document), fp)
        if VERBOSE: print('save as: ' + path)

        return "Saved Document " + document.document_id


    # only used for testing
    def save_to_collection_without_score(self, document, collection=None):
        VERBOSE = False

        # garantee that document is deleted from old datastore if 
        # annotation has reached a certain stage
        if "previous" in self.collections[collection]:
            to_remove = self.collections[collection]["previous"]
            for c in to_remove:
                self.remove_from_collection(document.document_id, c)
                        
        # save file to collection
        path = os.path.join(self.collections[collection]['path'], document.document_id+'.json')
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(JSONSerializable.serialize(document), fp)
        if VERBOSE: print('save as: ' + path)

        return "Saved Document " + document.document_id


    def load_from_datastore(self, document_id):
        for collection in self.collections.keys():
            if self.is_in_collection(document_id, collection):
                break
        path = os.path.join(self.collections[collection]['path'], document_id+'.json')
        with open(path, "r", encoding="utf-8") as fp:
            document = Document.deserialize(json.load(fp))
        return document

    def remove_from_collection(self, document_id, collection):
        # remove from scores file
        # if there are score_based_queues for this collection
        score_based_queues = self.queue_manager.get_score_based_queues(collection)
        if len(score_based_queues) > 0:
            # load score file
            document_scores = self.load_score_file(collection)
            
            # remove the scores
            if document_id in document_scores:
                del document_scores[document_id]
            
            # pickle score file
            path = os.path.join(self.collections[collection]['path'], 'scores')
            with open(path, 'wb') as fp:
                pickle.dump(document_scores, fp, protocol=pickle.DEFAULT_PROTOCOL)
        
        # remove document
        if self.is_in_collection(document_id, collection):
            path = os.path.join(self.collections[collection]['path'], document_id+'.json')
            os.remove(path)
        

    def get_document_importance_from_datastore(self, collection):
        # returns the document_id and the importance_score
        if os.path.isdir(self.collections[collection]['path']) == False:
            return []
        document_importance_list = []
        for document in document_generator(collection):
            document_importance_list.append({
                "document_id": document.document_id, 
                "importance_score": document.importance_score,
                "initial_importance": document.initial_importance,
                "annotated_by": document.annotated_by
            })
        return document_importance_list


    def document_generator(self, collection):
        # yields each document object in the datastore
        filelist = self.document_file_list(collection)
        total = len(filelist)
        interval_size = 1000
        t = time.time()
        #print("Started document generator", total)
        for idx, fname in enumerate(filelist):
            if idx % interval_size == 0:
                time_consumed = time.time() - t
                t = time.time()
                if time_consumed > 0:
                    fps = interval_size / time_consumed
                    remaining = total - idx
                    #print("Files per second: {:.2f} Done in: {:.2f}".format(fps, remaining / fps))
            
            path = os.path.join(self.collections[collection]['path'], fname)
            if not os.path.isfile(path):
                print("WARNING", path, "not found!")
                continue
            with open(path, "r", encoding="utf-8") as fp:
                document = Document.deserialize(json.load(fp))
            yield document
    
    def document_file_list(self, collection):
        filelist = os.listdir(self.collections[collection]['path'])
        filelist = [fname for fname in filelist if fname.endswith(".json")]
        return filelist
    
    def update_timestamps(self):
        # TODO: change function as needed
        # DO NOT CALL unless whole datastore is missing timestamps
        # update annotated_by timestamps for each collection
        for collection in self.collections:
            if collection == "other" or collection == "priority":
                continue
            # load filenames
            filelist = self.document_file_list(collection)
            # walk through all files
            for fname in filelist:
                path = os.path.join(self.collections[collection]['path'], fname)
                with open(path, "r", encoding="utf-8") as fp:
                    document = Document.deserialize(json.load(fp))
                # walk through annotation_types in annotated_by
                for annotation_type in document.annotated_by:
                    annotator_list = document.annotated_by[annotation_type]
                    if len(annotator_list) > 0:
                        # convert from 'annotator' to {name: 'annotator', timestamp: 21.04.2020}
                        new_annotator_list = []
                        for annotator in annotator_list:
                            updatedAnnotator = {}
                            updatedAnnotator["name"] = annotator
                            updatedAnnotator["timestamp"] = "2020-04-21T12:00:00.000Z"
                            new_annotator_list.append(updatedAnnotator)
                        # put updated list
                        document.annotated_by[annotation_type] = new_annotator_list
                # store document
                with open(path, "w", encoding="utf-8") as fp:
                    json.dump(JSONSerializable.serialize(document), fp)
                print("Updated document:", document.document_id, document.annotated_by)


def fname2document_id(fname):
    return fname[:len(fname)-len(".json")]






