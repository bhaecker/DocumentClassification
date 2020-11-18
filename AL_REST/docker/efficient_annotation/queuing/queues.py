from efficient_annotation.common import get_annotators
from itertools import count
from efficient_annotation.datastores import MongoDB
from efficient_annotation.logging import Logger
import os
import pickle
import json


class DocumentQueue:
    
    def __init__(self, datastore=None, name=None, follow_up_target_collection=None, weights = None):
        self.datastore = datastore
        self.name = name
        self.follow_up_target_collection = follow_up_target_collection
        self.weights = weights
        self.logger = Logger()
    
    def get_next(self, annotator_id = None):
        return next(self.generator)
    
    def create_generator(self, statistics):
        raise NotImplementedError
    
    def initialize_queue(self, statistics):
        self.generator = self.create_generator(statistics)


class CollectionListQueue(DocumentQueue):
    # Returns all documents from a list of collections.
    # Once all documents from one collection have been returned,
    # the queue moves on to the next collection in collection_order

    # if mix_collections = True the collections are treated as one collection
    # if False collections are shown in order of collection_order
    
    _ids = count(0)
    
    def __init__(self, collection_order=[], datastore=None, name=None, follow_up_target_collection=None, score_calculator = None, weights = None, mix_collections = False):
        if name == None:
            name = self.__class__.__name__ + "_" + next(self._ids)
        super().__init__(datastore=datastore, name=name, follow_up_target_collection=follow_up_target_collection)
        self.collection_order = collection_order
        self.score_calculator = score_calculator
        self.weights = weights
        self.mix_collections = mix_collections


        print("initialize CollectionListQueue with name " + str(name), flush=True)


    def initialize_queue(self, statistics):
        print("initialize_queue: CollectionListQueue " + self.name, flush=True)
        self.generator = self.create_generator(statistics)

        if self.score_calculator != None and isinstance(self.datastore, MongoDB):
            self.logger.log("add scores")
            self.datastore.add_scores(self.score_calculator.scoring_function, self.weights)

    
    def update_scores(self):
        if self.score_calculator != None and isinstance(self.datastore, MongoDB):
            self.logger.log("update scores")
            self.datastore.add_scores(self.score_calculator.scoring_function, self.weights)


    def create_generator(self, statistics):
        print("create CollectionListQueue generator for " + self.name, flush=True)

        if self.mix_collections: 
            print("CollectionListQueue show documents in collections: ", self.collection_order, flush = True)
            try:
                for document in self.datastore.document_generator_from_multiple_collections(self.collection_order, sort = True):
                    yield {
                        "document_id":document.document_id, 
                        "importance_score": document.importance_score, #document.initial_importance, 
                        "queue":self.name, 
                        "target":self.follow_up_target_collection[document.target_collection]} # target_collection is set to target by GUI
            except StopIteration:
                print("Finished ", str(self.collection_order), flush = True)
        else:
            for collection in self.collection_order:
                print("CollectionListQueue show documents in collection: ", collection, flush = True)
                try:
                    for document in self.datastore.document_generator(collection, sort = True):
                        yield {
                            "document_id":document.document_id, 
                            "importance_score": document.importance_score, #document.initial_importance, 
                            "queue":self.name, 
                            "target":self.follow_up_target_collection[document.target_collection]} # target_collection is set to target by GUI
                except StopIteration:
                    print("Finished ", collection, flush = True)



# returns documents that match a query in the mongodb
class QueryQueue(DocumentQueue):

    _ids = count(0)
    
    def __init__(self, queries = None, datastore=None, name=None, follow_up_target_collection=None, score_calculator = None, weights = None, mix_collections = False):
        if name == None:
            name = self.__class__.__name__ + "_" + next(self._ids)
        super().__init__(datastore=datastore, name=name, follow_up_target_collection=follow_up_target_collection)
        self.queries = queries
        self.score_calculator = score_calculator
        self.weights = weights
        self.mix_collections = mix_collections

        print("initialize QueryQueue with name " + str(name), flush=True)


    def initialize_queue(self, statistics):
        print("initialize_queue: CollectionListQueue " + self.name, flush=True)
        self.generator = self.create_generator(statistics)

        if self.score_calculator != None and isinstance(self.datastore, MongoDB):
            self.logger.log("add scores")
            self.datastore.add_scores(self.score_calculator.scoring_function, self.weights)

    
    def update_scores(self):
        if self.score_calculator != None and isinstance(self.datastore, MongoDB):
            self.logger.log("update scores")
            self.datastore.add_scores(self.score_calculator.scoring_function, self.weights)


    def create_generator(self, statistics):

        for query in self.queries:
            self.logger.log("QueryQueue show documents that match query: " + str(query))
            try:
                cursor = self.datastore.document_generator_from_query(query, sort = False)
                for document in cursor:
                    
                    if self.follow_up_target_collection == None:
                        follow_up = document.target_collection
                    else: 
                        follow_up = self.follow_up_target_collection[document.target_collection]

                    # add document id to blocked list
                    if document != None:
                        self.datastore.add_id(document.document_id)

                    yield {
                        "document_id":document.document_id, 
                        "importance_score": document.importance_score, #document.initial_importance, 
                        "queue":self.name, 
                        "target":follow_up} # target_collection is set to target by GUI

                cursor.close() # cursor needs to be closed because we use no_cursor_timeout=True in find (see mongodb.py document_generator_from_query)
                self.logger.log("Finished: no more documents in queue " + str(self.name))

            except StopIteration:
                self.logger.log("Finished: show documents that matches query: " + str(query))


# simple queue without double/triple annotation
# uses annotation vectors of base model segment_multimodal_model
# get_next returns a document from a different predicted class 
# warning: currently implemented to use only first item in collection_order
class DiversityQueue(DocumentQueue):
    
    _ids = count(0)
    
    def __init__(self, collection_order=[], datastore=None, name=None, follow_up_target_collection=None, score_calculator = None, weights = None):
        if name == None:
            name = self.__class__.__name__ + "_" + next(self._ids)
        super().__init__(datastore=datastore, name=name, follow_up_target_collection=follow_up_target_collection)
        self.collection_order = collection_order
        self.score_calculator = score_calculator
        self.weights = weights
        self.number_document_label_classes = len(self.datastore.annotation_types["document_label"]["labels"])
        self.document_label_counter = [0] * self.number_document_label_classes
        self.desired_label = 0 

        print("initialize DiversityQueue with name " + str(name), flush=True)


    def initialize_queue(self, statistics):
        print("initialize_queue: DiversityQueue " + self.name, flush=True)
        self.generator = self.create_generator(statistics)

        if self.score_calculator != None and isinstance(self.datastore, MongoDB):
            print("add scores")
            self.datastore.add_scores(self.score_calculator.scoring_function, self.weights)

        self.datastore.add_document_label_predicted_by_base_model("ml_out") # ADDED


    def get_next(self, annotator_id = None):

        if self.desired_label >= (self.number_document_label_classes):
            self.desired_label = 0

        print("get next for class label", self.desired_label)
        
        #document_json = self.datastore.get_document_with_predicted_document_label(self.collection_order[0], self.desired_label)
        document_json = self.datastore.get_document_with_predicted_document_label_v2(self.collection_order[0], self.desired_label)

        tries = 0 
        # try until a document in one of the document classes is found
        # if there is no document in any of the document classes -> return None
        while tries <= self.number_document_label_classes:
            if (document_json != None):
                for document_label_annotation in document_json["document_label"]["annotations"]:
                    if document_label_annotation["annotator_id"] == "segment_multimodal_model":
                        #vec = document_label_annotation["annotation_vector"]
                        #maximum = max(vec)
                        #index = [i for i, j in enumerate(vec) if j == maximum] 
                        
                        #print("vec", vec)
                        #print("maximum", maximum)
                        #print("index", index)
        
                        self.document_label_counter[self.desired_label] += 1
                        self.logger.log("show document with document_label: " + str(self.desired_label),"queue_log.txt")
                        self.logger.log("document_label_counter: " + str(self.document_label_counter), "queue_log.txt")
                        self.logger.log("sum document_label_counter: " + str(sum(self.document_label_counter)), "queue_log.txt")


                        self.desired_label += 1

                        #print(document_json, flush = True)
                        return {
                        "document_id":document_json["document_id"], 
                        "importance_score": document_json["importance_score"], 
                        "queue":self.name, 
                        "target":self.follow_up_target_collection[document_json["target_collection"]]}
            else:
                self.desired_label += 1
                if self.desired_label >= (self.number_document_label_classes):
                    self.desired_label = 0
                    
                print("try again for new class label ", self.desired_label, flush = True)

                tries += 1

                
                #document_json = self.datastore.get_document_with_predicted_document_label(self.collection_order[0], self.desired_label)
                document_json = self.datastore.get_document_with_predicted_document_label_v2(self.collection_order[0], self.desired_label)


        return None # if no next documents can be found

        
    
    def create_generator(self, statistics):
        print("")
        # print("create CollectionListQueue generator for " + self.name, flush=True)

        # for collection in self.collection_order:
        #     print("CollectionListQueue show documents in collection: ", collection, flush = True)
        #     try:
        #         for document in self.datastore.document_generator(collection):
        #             yield {
        #                 "document_id":document.document_id, 
        #                 "importance_score": document.importance_score, #document.initial_importance, 
        #                 "queue":self.name, 
        #                 "target":self.target} # target is the target_collection specified by the queue
        #     except StopIteration:
        #         print("Finished ", collection, flush = True)



class MultiAnnotatorQueue(DocumentQueue):

    _ids = count(0)
    
    def __init__(self, get_annotator_fn=None, origin_collection=None, datastore=None, name=None, follow_up_target_collection=None, weights = None):
        if name == None:
            name = self.__class__.__name__ + "_" + next(self._ids)
        super().__init__(datastore=datastore, name=name, follow_up_target_collection=follow_up_target_collection, weights=weights)
        self.origin_collection = origin_collection
        self.get_annotator_fn = get_annotator_fn
        print("initialize MultiAnnotatorQueue with name " + str(name), flush=True)


    def create_generator(self, statistics):
        print("create MultiAnnotatorQueue generator for " + self.name, flush=True)

        # yield docs with no annotation
        docs_without_human_annotation = self.datastore.get_documents_without_human_annotation(self.origin_collection)
        for doc in docs_without_human_annotation:
            yield {
                "document_id":doc["document_id"], 
                "importance_score": doc["importance_score"], 
                "queue":self.name, 
                "target":self.follow_up_target_collection[doc["target_collection"]]}
        


# Anna: updated queue
# can handle one or multiple annotators
# TODO all queues types should use this queue (queues with no score, with only one collection etc.)
# collection_order ... gives the order in which documents should be put in queue
# e.g. collection_order = ["priority", "other"] in queue there will first be documents from priority
# ordered by importance_score followed by documents from other ordered by importance_score
# TODO based on the current collection the queue decides the suitable score calculator (a mapping could be given by manager)
class ScoreBasedMultiAnnotatorQueue(DocumentQueue):
    
    _ids = count(0)
    
    def __init__(self, score_calculator=None, collection_order = None, datastore=None, name=None, follow_up_target_collection=None, weights=None):
        if name == None:
            name = self.__class__.__name__ + "_" + next(self._ids)
        super().__init__(datastore=datastore, name=name, follow_up_target_collection=follow_up_target_collection)
        
        print("initialize ScoreBasedMultiAnnotatorQueue with name " + str(name), flush=True)

        # score_calculator that has scoring_function and weight_function
        self.score_calculator = score_calculator
        self.collection_order = collection_order
        self.follow_up_target_collection = follow_up_target_collection
        self.weights = weights

        # TODO might give scores only to docs with from some collection to save time 
        # TODO choose score_calculator based on collection?
        if score_calculator != None and isinstance(self.datastore, MongoDB):
            print("add scores with weights")
            self.datastore.add_scores(self.score_calculator.scoring_function, self.weights)

        # this generator/queue is shared by all annotators
        self.generator_for_docs_without_human_annotations = self.create_generator(None)
        self.annotators = []

        # these generators/queues are specific to one annotator
        self.generators = {} # annotator_id:generator

    # reset queue
    def initialize_queue(self, statistics = None):

        if self.score_calculator != None and isinstance(self.datastore, MongoDB):
            self.datastore.add_scores(self.score_calculator.scoring_function, self.weights)

        # this generator/queue is shared by all annotators
        self.generator_for_docs_without_human_annotations = self.create_generator(self.weights, None)
        self.annotators = []

        # these generators/queues are specific to one annotator
        self.generators = {} # annotator_id:generator


    # TODO if we have performance issues: annotator specific generators only have to be created if
    # generator_for_docs_without_human_annotations is empty
    def add_annotator(self, annotator_id):
        if annotator_id not in self.annotators:
            print("add annotator " + str(annotator_id))
            self.annotators.append(annotator_id)
            generator = self.create_generator(self.weights, annotator_id)
            self.generators[annotator_id] = generator

    def get_next(self, annotator_id = None):

        self.add_annotator(annotator_id)

        # mongodb will return sorted documents sorted
        try:
            next_doc = next(self.generator_for_docs_without_human_annotations)
            #print("next_doc:" + str(next_doc), flush = True)
        except StopIteration:
            print("#### no more documents without complete annotations ####")
            # if no unannotated is available, get document not seen by current annotator
            next_doc = next(self.generators[annotator_id])
        return next_doc
        
    # create a generator for a specific collection (queue status)
    def create_generator(self, statistics, annotator_id = None):
        print("create ScoreBasedMultiAnnotatorQueue generator for " + self.name, flush=True)

        # create shared generator for documents without human annotations
        # create generator for specific annotators only when this generator is empty
        if (annotator_id == None):
            # yield docs with no annotation
            print("1. get unannotated docs")
            docs_without_human_annotation = self.datastore.get_documents_without_human_annotation(self.collection_order)
            for doc in docs_without_human_annotation:
                yield {
                    "document_id":doc["_id"], 
                    "importance_score": doc["importance_score"], 
                    "queue":self.name, 
                    "target":self.follow_up_target_collection[doc["target_collection"]]}  # based on current target_collection choose next target collection

            
            docs_without_complete_human_annotation = self.datastore.get_documents_without_complete_human_annotation()
            print("2. get documents with partial but without complete human annotation")
            # get documents that have partial annotation but no complete annotation
            for doc in docs_without_complete_human_annotation:
                #print("yield: docs_without_complete_human_annotation")
                #print("doc target_collection  = " + str(doc["target_collection"]))
                #print("doc id  = " + str(doc["_id"]))

                yield {
                    "document_id":doc["_id"], 
                    "importance_score": doc["importance_score"], 
                    "queue":self.name, 
                    "target":self.follow_up_target_collection[doc["target_collection"]]} # based on current target_collection choose next target collection

        else: 
            # get all docs that current annotator has not seen yet
            # first: docs with one previous annotator (in complete)
            # then: docs with two previous annotators (in complete)

            for num_annotators in [1,2]: 
                print("3. get docs with complete annotation but not seen by this annotator")
                docs_not_annotated_by = self.datastore.get_document_importance_from_datastore_not_annotated_by(num_annotators, annotator_id, self.collection_order)
                for doc in docs_not_annotated_by:
                    yield {
                        "document_id":doc["_id"], 
                        "importance_score": doc["importance_score"], 
                        "queue":self.name, 
                        "target":self.follow_up_target_collection[doc["target_collection"]]} # based on current target_collection choose next target collection

        
class DoubleQueue(DocumentQueue):
    # a queue that picks from elements from one of two queues
    # based on some condition function
    # If the function returns true, pick queue1 otherwise pick queue2
    # if either of the two runs out of documents
    # raises a StopIteration
    #
    # WARNING:
    # The collections from which queues take their documents MUST be different,
    # otherwise inconsistencies can happen.
    
    _ids = count(0)
    
    def __init__(self, queue1, queue2, reinitialize1=None, reinitialize2=None, condition_fn=lambda : True, datastore=None, name=None, follow_up_target_collection=None):
        if name == None:
            name = self.__class__.__name__ + "_" + next(self._ids)
        super().__init__(datastore=datastore, name=name, follow_up_target_collection=follow_up_target_collection)
        self.queue1 = queue1
        self.reinitialize1 = reinitialize1
        self.queue2 = queue2
        self.reinitialize2 = reinitialize2
        self.condition_fn = condition_fn
        print("initialize DoubleQueue", flush=True)
        
    
    # call next, also auto-reinitialize
    def get_next(self, annotator_id = None):
        if self.condition_fn():
            try:
                print("get_next in DoubleQueue for queue1:" + self.queue1.name, flush=True)

                return self.queue1.get_next(annotator_id)
            except StopIteration:
                if self.reinitialize1:
                    self.queue1.initialize_queue(self.reinitialize1())
                    return self.queue1.get_next(annotator_id)
                else:
                    raise StopIteration
        else:
            try:
                print("get_next in DoubleQueue for queue2:" + self.queue2.name, flush=True)
                return self.queue2.get_next(annotator_id)
            except StopIteration:
                if self.reinitialize2:
                    self.queue2.initialize_queue(self.reinitialize2())
                    return self.queue2.get_next(annotator_id)
                else:
                    raise StopIteration
        
    
    def initialize_queue(self, statistics):
        self.queue1.initialize_queue(statistics)
        self.queue2.initialize_queue(statistics)
