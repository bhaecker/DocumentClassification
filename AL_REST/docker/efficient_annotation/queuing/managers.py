import os
import time
import sys
from efficient_annotation.importance_scores import DiversityScoreCalculator, InitialScoreCalculator, AnnotationVectorScoreCalculator
from efficient_annotation.importance_scores import entropy_fn, random_fn, least_confident_fn, margin_sampling_fn
#from efficient_annotation.queuing import DocumentQueue, CollectionListQueue, ScoreBasedQueue, MultiAnnotatorQueue, ScoreBasedMultiAnnotatorQueue, DoubleQueue
from efficient_annotation.queuing import DocumentQueue, CollectionListQueue, DiversityQueue, MultiAnnotatorQueue, ScoreBasedMultiAnnotatorQueue, DoubleQueue, QueryQueue
from efficient_annotation.logging import Logger
from efficient_annotation.common import load_config_file


class QueueManager:
    
    def __init__(self, datastore=None, annotation_types={}):
        config = load_config_file()

        self.DEBUG_switch_to_partial_after_X_documents = config["debug"]["DEBUG_switch_to_partial_after_X_documents"] 

        print("initialize queue manager", flush=True)

        self.annotation_types = annotation_types
        # Datastore
        self.ds = datastore
        self.ds.register_queue_manager(self)
        # current queue
        self.current_queue = None

        # statistics
        self.statistics = None

        self.logger = Logger()

        # TODO: when should the document be put in total_annotation 
        self.follow_up_target_collection = {
            "priority": "partial",
            "other": "partial",
            "partial": "complete",
            "complete": "complete", # will be shown again for 2nd and 3rd annotation automatically
            "ml_out": "partial",
            "total_priority": "total_annotation",
            "2nd_annotation": "3rd_annotation",
            "3rd_annotation": "total_annotation", 
        }

        self.follow_up_target_collection_complete_to_2nd = {
            "priority": "partial",
            "other": "partial",
            "partial": "complete",
            "complete": "2nd_annotation",
            "ml_out": "partial",
            "total_priority": "total_annotation",
            "2nd_annotation": "3rd_annotation",
            "3rd_annotation": "total_annotation", 
        }



        # used during annotation session:
        # count number of partially labeled docs - stop at 5000 partially labeled docs
        self.number_of_partially_labeled_docs = self.ds.get_number_of_documents("has_partial_human_annotation")
        
        if self.DEBUG_switch_to_partial_after_X_documents != 5000:
            self.logger.log("WARNING: DEBUG queue switch (al_partial to al_complete_show_new_docs). Switch to al_complete_show_new_docs after " + str(self.DEBUG_switch_to_partial_after_X_documents) + " documents.")
            self.max_number_of_partially_labeled_docs = self.DEBUG_switch_to_partial_after_X_documents
        else:
            self.max_number_of_partially_labeled_docs = 5000

        self.logger.log("Starting up queue manager")
        self.logger.log("number_of_partially_labeled_docs: " + str(self.number_of_partially_labeled_docs), "queue_log.txt")

        # registered queues
        self.queues = {
            "initial": CollectionListQueue( 
                collection_order = ["priority", "other"], 
                datastore=self.ds, 
                name="initial", 
                follow_up_target_collection = {"priority": "partial", "other": "partial"},
                mix_collections = False
            ),
            "partial": ScoreBasedMultiAnnotatorQueue ( # was ScoreBasedQueue
                score_calculator=DiversityScoreCalculator(
                    annotation_types=self.annotation_types, important_types=["segment_label", "segment_type"]), 
                collection_order = ["partial"], 
                datastore=self.ds, 
                name="partial", 
                follow_up_target_collection = self.follow_up_target_collection_complete_to_2nd),
            "complete": CollectionListQueue(
                collection_order = ["complete", "2nd_annotation", "3rd_annotation", "total_annotation"],
                datastore=self.ds, 
                name="complete", 
                follow_up_target_collection = self.follow_up_target_collection_complete_to_2nd,
                mix_collections = False

            ), # changed from complete
            "2nd_annotation": ScoreBasedMultiAnnotatorQueue( # TODO remove this queue if we do not need specifically docs with 2 annotators
                score_calculator=InitialScoreCalculator(), 
                collection_order = ["complete"],
                datastore=self.ds, 
                name="2nd_annotation", 
                follow_up_target_collection = self.follow_up_target_collection_complete_to_2nd), # TODO should be 3rd_annotation?
            "3rd_annotation": ScoreBasedMultiAnnotatorQueue( # was MultiAnnotatorQueue  # TODO remove this queue if we do not need specifically docs with 2 annotators
                collection_order = ["2nd_annotation", "complete"], # TODO only return docs for 3rd_annotation
                datastore=self.ds, 
                name="3rd_annotation", 
                follow_up_target_collection = self.follow_up_target_collection_complete_to_2nd),
            "total_priority": ScoreBasedMultiAnnotatorQueue(
                score_calculator=InitialScoreCalculator(), 
                collection_order = ["complete"], 
                datastore=self.ds, 
                name="total_priority", 
                follow_up_target_collection = self.follow_up_target_collection),
            "total": CollectionListQueue( # TODO what is the purpose of this? can use only total_priority
                collection_order = ["complete"], 
                datastore=self.ds, 
                name="total", 
                follow_up_target_collection = {"complete": "total_annotation"},
                mix_collections = False
            ),
            "al_entropy": ScoreBasedMultiAnnotatorQueue(  
                score_calculator=AnnotationVectorScoreCalculator(
                    annotation_types=self.annotation_types,
                    important_types=["document_label", "document_type", "segment_label", "segment_type"],
                    dist_scoring_fn=entropy_fn), 
                collection_order = ["ml_out", "complete"], 
                datastore=self.ds, 
                name="al_entropy", 
                follow_up_target_collection = self.follow_up_target_collection,
                weights=self.get_statistics),
            # "ml_out_diversity": ScoreBasedMultiAnnotatorQueue(  
            #     score_calculator=DiversityScoreCalculator(
            #         annotation_types=self.annotation_types, important_types=["segment_label", "segment_type"]), 
            #     collection_order = ["ml_out", "complete"], 
            #     datastore=self.ds, 
            #     name="ml_out_diversity", 
            #     follow_up_target_collection = self.follow_up_target_collection,
            #     weights=self.get_statistics),
            "al_random": ScoreBasedMultiAnnotatorQueue ( 
                score_calculator=AnnotationVectorScoreCalculator(
                    annotation_types=self.annotation_types,
                    important_types=["document_label", "document_type", "segment_label", "segment_type"],
                    dist_scoring_fn=random_fn), 
                collection_order = ["ml_out", "complete"], 
                datastore=self.ds, 
                name="al_random", 
                follow_up_target_collection = self.follow_up_target_collection,
                ),
            # "al_margin_sampling": ScoreBasedMultiAnnotatorQueue ( 
            #     score_calculator=AnnotationVectorScoreCalculator(
            #         annotation_types=self.annotation_types,
            #         important_types=["document_label", "document_type", "segment_label", "segment_type"],
            #         dist_scoring_fn=margin_sampling_fn), 
            #     collection_order = ["ml_out", "complete"], 
            #     datastore=self.ds, 
            #     name="al_margin_sampling", 
            #     follow_up_target_collection = self.follow_up_target_collection,
            #     ),            
            # "al_least_confident": ScoreBasedMultiAnnotatorQueue ( 
            #     score_calculator=AnnotationVectorScoreCalculator(
            #         annotation_types=self.annotation_types,
            #         important_types=["document_label", "document_type", "segment_label", "segment_type"],
            #         dist_scoring_fn=least_confident_fn), 
            #     collection_order = ["ml_out", "complete"], 
            #     datastore=self.ds, 
            #     name="al_least_confident", 
            #     follow_up_target_collection = self.follow_up_target_collection,
            #     ),

            ### queues for annotation session ###

            "al_view": DiversityQueue(  
                score_calculator=None, 
                collection_order = ["ml_out"], 
                datastore=self.ds, 
                name="al_view", 
                follow_up_target_collection = {"ml_out": "viewed"},
                weights=None),

            # al_view_finetuned: queue to show documents that have finetuned predictions (has_finetuned = True)
            "al_view_finetuned": QueryQueue(  
                score_calculator=None, 
                queries = [
                    # documents with predictions from finetuned model
                    {"$and": [
                        {"has_finetuned":True},
                    ]},
                    # documents with predictions from base model with complete human predictions
                    {"$and": [
                        {"has_finetuned":False},
                        {"has_base":True},
                        {"target_collection":"has_complete_human_annotation"}
                    ]},
                    # documents with predictions from base model with partial human predictions
                    {"$and": [
                        {"has_finetuned":False},
                        {"has_base":True},
                        {"target_collection":"has_partial_human_annotation"}
                    ]},
                    # documents with only base predictions and no human annotations
                    {"$and": [
                        {"has_finetuned":False},
                        {"target_collection": "ml_out"}
                        #{"$and": [{"target_collection": {"$not":{"has_partial_human_annotation"}}}, {"target_collection":{"$not":{"has_complete_human_annotation"}}}]},
                    ]},
                    # documents from training, target collections:
                    #3rd_annotation, 2nd_annotation, total_annotation, complete, partial, initial, other
                    {"target_collection":"3rd_annotation"},
                    {"target_collection":"2nd_annotation"},        
                    {"target_collection":"total_annotation"},        
                    {"target_collection":"complete"},        
                    {"target_collection":"partial"},        
                    {"target_collection":"initial"},        
                    {"target_collection":"other"}        
                    ],
                datastore=self.ds, 
                name="al_view_finetuned", 
                follow_up_target_collection = None, # None: target is set to current target_collection
                weights=None),

            "al_partial": DiversityQueue(  
                score_calculator=None, 
                collection_order = ["ml_out"], 
                datastore=self.ds, 
                name="al_partial", 
                follow_up_target_collection = {"ml_out": "has_partial_human_annotation"},
                weights=None),

            "al_complete_show_new_docs": DiversityQueue(  
                score_calculator=None, 
                collection_order = ["ml_out"], 
                datastore=self.ds, 
                name="al_complete_show_new_docs", 
                follow_up_target_collection = {"ml_out": "has_complete_human_annotation"},
                weights=None),

            "al_complete": QueryQueue(  
                score_calculator=None, 
                queries = [
                    # show again documents from target_collection has_partial_human_annotation
                    {"$and": [
                    {"target_collection":"has_partial_human_annotation"},
                    {"document_id" : { "$nin": self.ds.list_of_blocked_ids }} 
                    ]}
                    ],
                datastore=self.ds, 
                name="al_complete", 
                follow_up_target_collection = {"has_partial_human_annotation": "has_complete_human_annotation"},
                weights=None),


            # "al_partial": ScoreBasedMultiAnnotatorQueue(  
            #     score_calculator=AnnotationVectorScoreCalculator(
            #         annotation_types=self.annotation_types,
            #         important_types=["document_label", "document_type", "segment_label", "segment_type"],
            #         dist_scoring_fn=entropy_fn), 
            #     collection_order = ["ml_out"], 
            #     datastore=self.ds, 
            #     name="al_partial", 
            #     follow_up_target_collection = {"ml_out": "has_partial_human_annotation"},
            #     weights=self.get_statistics),

            # "al_complete_show_new_docs": ScoreBasedMultiAnnotatorQueue(  
            #     score_calculator=AnnotationVectorScoreCalculator(
            #         annotation_types=self.annotation_types,
            #         important_types=["document_label", "document_type", "segment_label", "segment_type"],
            #         dist_scoring_fn=entropy_fn), 
            #     collection_order = ["ml_out"], 
            #     datastore=self.ds, 
            #     name="al_complete_show_new_docs", 
            #     follow_up_target_collection = {"ml_out": "has_complete_human_annotation"},
            #     weights=self.get_statistics),

            ### queues for evaluation ###
            "eval_random": CollectionListQueue(
                collection_order = ["has_complete_human_annotation", "has_partial_human_annotation"], 
                datastore=self.ds, 
                name="eval_random", 
                follow_up_target_collection = {"has_complete_human_annotation": "has_complete_human_annotation", "has_partial_human_annotation": "has_partial_human_annotation"},
                score_calculator = AnnotationVectorScoreCalculator(
                    annotation_types=self.annotation_types,
                    important_types=["document_label", "document_type", "segment_label", "segment_type"],
                    dist_scoring_fn=random_fn), 
                weights = self.get_statistics,
                mix_collections = True
                ),

            # TODO now uses entropy, could be changed to diversity
            "eval_al": CollectionListQueue(
                collection_order = ["has_complete_human_annotation", "has_partial_human_annotation"], 
                datastore=self.ds, 
                name="eval_al", 
                follow_up_target_collection = {"has_complete_human_annotation": "has_complete_human_annotation", "has_partial_human_annotation": "has_partial_human_annotation"},
                score_calculator = AnnotationVectorScoreCalculator(
                    annotation_types=self.annotation_types,
                    important_types=["document_label", "document_type", "segment_label", "segment_type"],
                    dist_scoring_fn=entropy_fn), 
                weights = self.get_statistics,
                mix_collections = True
                ),
                
        }
        # composite queues
        main_queue_1 = self.queues["3rd_annotation"]
        main_queue_2 = DoubleQueue(self.queues["2nd_annotation"], self.queues["partial"], reinitialize1=self.get_statistics, condition_fn=self.second_annotation_condition, name="main_2nd_annotation")
        main_queue = DoubleQueue(main_queue_1, main_queue_2, reinitialize1=self.get_statistics, condition_fn=self.third_annotation_condition, name="main")
        self.queues["main_2nd_annotation"] = main_queue_2
        self.queues["main"] = main_queue
        
        self.follow_up_queue = {
            "initial": "main",
            "main": "initial",
            "complete": "complete",
            "total_priority": "total",
            "total": "main",
            "al_entropy": None,
            "al_random": None,
            "ml_out_diversity": None,
            "al_partial": "al_complete",
            "eval_al": None,
            "eval_random": None,
            "al_view": None,
            "al_complete": "al_complete"
        }
        
        self.valid_queue_types = [] 
        
        for type in self.queues:
            self.valid_queue_types.append(type)

        
        # current_annotator
        self.current_annotator = None
        
        # active flags
        self.active_flags = set()
    
    
    ##########################################################################
    ## Start second and third iteration only, if there is at least a buffer 
    ## of at least some documents
    def second_annotation_condition(self):
        # Returns true if second annotation should be performed
        # Second annotation should be performed on 15% of documents
        count_single_annotation = self.ds.get_number_of_documents("complete")#len(self.ds.document_file_list("complete"))
        count_second_annotation = self.ds.get_number_of_documents("2nd_annotation") #len(self.ds.document_file_list("2nd_annotation"))
        total_count = count_single_annotation + count_second_annotation
        # buffer
        if count_single_annotation >= 10:
            self.active_flags.add("2nd_annotation")
        if "2nd_annotation" not in self.active_flags:
            return False
        if count_single_annotation == 0 or total_count == 0:
            return False
        return count_second_annotation / total_count < 0.15
    
    def third_annotation_condition(self):
        # Returns true if third annotation should be performed
        # Third annotation should be performed on 25 documents
        count_second_annotation = self.ds.get_number_of_documents("2nd_annotation") 
        count_third_annotation = self.ds.get_number_of_documents("3rd_annotation") 
        # buffer
        if count_second_annotation >= 10:
            self.active_flags.add("3rd_annotation")
        if "3rd_annotation" not in self.active_flags:
            return False
        return count_third_annotation < 25
    
    def get_annotator_fn(self):
        return self.current_annotator
    
    def total_annotation_priority_condition(self):
        return self.ds.get_number_of_documents("total_annotation")  < 50
    
    def get_score_based_queues(self, collection):
        # returns all queues associated with a collection
        score_based_queues = []
        for q in self.queues.values():
            if (isinstance(q, ScoreBasedQueue) or isinstance(q, ScoreBasedMultiAnnotatorQueue)) and q.origin_collection == collection:
                score_based_queues.append(q)
        return score_based_queues
    
    
    def update_statistics(self, statistics):
        self.statistics = statistics
    
    def get_statistics(self):
        return self.statistics

    def reset_partial_queue_counter(self):
        self.logger.log("reset_partial_queue_counter to 0")
        self.number_of_partially_labeled_docs = 0
    
    def get_next(self, annotator_id):
        # Get the next document in the document queue
        # If the queue is out of documents, switch queue_type and create a new document queue
        try:
            self.current_annotator = annotator_id

            if self.current_queue.name == "al_partial": 

                self.number_of_partially_labeled_docs = self.ds.get_number_of_documents("has_partial_human_annotation")

                self.logger.log("al_partial. number_of_partially_labeled_docs: " + str(self.number_of_partially_labeled_docs), "queue_log.txt")
                if self.number_of_partially_labeled_docs >= self.max_number_of_partially_labeled_docs:
                    self.logger.log("al_partial: switch", "queue_log.txt")
                    raise StopIteration

            next_document = self.current_queue.get_next(annotator_id)

            if self.current_queue.name == "al_complete":  
                # check if next_document fits criteria
                # else: get new next_document
                suitable_doc_found = False
                idx_of_valid_document_label_classes = [0,1,2,3,4,5,6,8,11,13,14,28,30]
                
                valid_is_other_document_label_classes = ["Technischer Plan", "technischer Plan", "Technischer Plan ", "Plan", "Zeichnung", "Bauplan", "Technische Zeichnung", "technische Zeichnung", "Bauskizze", "Architektenplan"
					"Prüfbericht", "Pr\u00fcfbericht", "Prüfbericht ", "Gutachten", "Untersuchungsbericht",
					"Mitarbeiterinformation", "Bewerbungsschreiben",
					"Anwaltsschreiben",
					"Einladung",
					"Anleitung",
					"AGBS",
					"is_other", "Sonstiges"]


                while suitable_doc_found == False:
                    if  next_document == None:
                        self.logger.log("al_complete: next_document is None -> raise StopIteration")
                        raise StopIteration

                    if next_document == "{}" or next_document == {}:
                        self.logger.log("WARNING: next doc is empty")


                    full_doc = self.ds.load_from_datastore(next_document["document_id"])

                    if full_doc == "{}" or full_doc == {}:
                        self.logger.log("WARNING: no doc with id" + str(next_document["document_id"])+ " found.")

                    for annotation in full_doc.document_label.annotations:
                        if annotation.annotator_type == "human":
                        # look for index of 1
                            if ((1 in annotation.annotation_vector and annotation.annotation_vector.index(1) in idx_of_valid_document_label_classes) 
                            or annotation.is_other in valid_is_other_document_label_classes):
                                suitable_doc_found = True
                                self.logger.log("al_complete: return document " + str(next_document["document_id"]))

                                if 1 in annotation.annotation_vector:
                                    self.logger.log("index (finetuned annotation types): " + str(annotation.annotation_vector.index(1)))

                                if annotation.is_other != "":
                                    self.logger.log("annotation.other: " +str(annotation.is_other))

                            else:
                                self.logger.log("document did not fit criteria. get next doc")
                                next_document = self.current_queue.get_next(annotator_id)


        except StopIteration:
            self.logger.log("WARNING: managers: StopIteration occured.")

            self.active_flags = set()
            new_queue_name = self.follow_up_queue[self.current_queue.name]
            if new_queue_name == None:
                return 
            
            # try to return next document with new queue instead
            self.logger.log("switch to queue: " + str(new_queue_name), "queue_log.txt")

            self.initialize_queue(new_queue_name)
            next_document = self.current_queue.get_next(annotator_id)

        self.logger.log("next_document: " + str(next_document), "queue_log.txt")
        self.logger.log("annotator_id: " + str(annotator_id), "queue_log.txt")
        self.logger.log("next_document: " + str(next_document))
        self.logger.log("annotator_id: " + str(annotator_id))

        return next_document
    
    def initialize_queue(self, queue_type):
        # Initialize the queue with a new queue_type
        if queue_type not in self.valid_queue_types:
            print("Queue Manager: Not a valid queue type. Please set queue_type to one of " + str(self.valid_queue_types), flush=True)
            return 'Not a valid queue type. Please set queue_type to one of ' + str(self.valid_queue_types)

        if queue_type == "al_partial" or queue_type == "al_complete_show_new_docs" or queue_type == "al_complete":
            self.logger.log("Reset viewed documents to target_collection ml_out", "queue_log.txt")
            self.ds.set_target_collection_old_to_new("viewed", "ml_out")

        self.current_queue = self.queues[queue_type]
        self.current_queue.initialize_queue(self.statistics)

        self.logger.log("Setting current_queue to: " + queue_type, "queue_log.txt")
        return "Setting current_queue to " + queue_type

    def get_current_queue(self):
        return self.current_queue

    def update_scores_in_current_queue(self):
        return_message = ""
        if (isinstance(self.current_queue, CollectionListQueue)): # TODO implement update_scores for other queue types
            self.current_queue.update_scores()
            return_message = "Done: update scores in current queue."
        else:
            return_message = "WARNING: could not update scores in current queue. current queue is not instance of CollectionListQueue."

        return return_message

