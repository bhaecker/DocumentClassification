from copy import deepcopy
import threading
import time

from efficient_annotation.common import get_human_annotation

class StatisticsCalculator:
    # Calculates the annotation statistics of a particular datastore
    # Updates are calculated in a separate thread
    # While the update is running, the previously calculated statistics are returned
    
    def __init__(self, datastore, config, annotation_types):
        self.datastore = datastore
        self.config = config
        self.annotation_types = annotation_types
        self.expected = config['statistics']['expected']
        # statistics thread + object
        # only the statistics_thread must modify the statistics object
        self.statistics_thread = None
        self.statistics_object = self.__create_statistics(Counter(self.annotation_types))
        # lock to control access to variable
        self.statistics_thread_datalock = threading.Lock()
        self.statistics_object_datalock = threading.Lock()
        
    
    def get_statistics(self):
        # Make sure that the statistics_object is not modified during read
        with self.statistics_object_datalock:
            stats = deepcopy(self.statistics_object)
        return stats
    
    def __create_statistics(self, counter):
        statistics = deepcopy(counter.counts)
        for key, value in self.annotation_types.items():
            for label in value["labels"]:
                statistics[key][label] = [statistics[key][label], self.expected]
        return statistics
    
    def get_segment_annotation_counter(self, segment):
        # Returns annotation counts for each label in the segment
        # Repeats for child segments
        segment_annotation_counter = Counter(self.annotation_types)
        # Get and add labels for all groups in this segment
        for annotation_type, group in segment.annotation_groups.items():
            if annotation_type == "segment_boundary":
                continue
            if annotation_type in self.annotation_types:
                label = get_human_annotation(group, self.annotation_types[annotation_type]['labels'])
                segment_annotation_counter.count(annotation_type, label)
        # Do the same for all child segments and add to this count
        for child_segment in segment.child_segments:
            child_counter = self.get_segment_annotation_counter(child_segment)

            # TODO what is this? add_counters does not exist
            segment_annotation_counter = add_counters(segment_annotation_counter, child_counter)
        # Return total
        return segment_annotation_counter
    
    def update_counter(self, counter, collection):
        #if os.path.isdir(self.config['collections'][collection]['path']) == False:
        #    print("No documents in", collection)
        #    return counter

        # go through each document
        for document in self.datastore.document_generator(collection):
            # collect document label annotations
            document_label = get_human_annotation(document.document_label, self.annotation_types["document_label"]['labels'])
            counter.count("document_label", document_label)
            
            # collect document type annotations
            document_type = get_human_annotation(document.document_type, self.annotation_types["document_type"]['labels'])
            counter.count("document_type", document_type)
            
            # collect document_contains if available
            if len(document.document_contains.annotations) > 0:
                vector = document.document_contains.annotations[0].annotation_vector
                labels = self.annotation_types["document_contains"]["labels"]
                for label, entry in zip(labels, vector):
                    counter.count("document_contains", label, amount=entry)
            
            # go through each page
            for page in document.pages:
                # go through each segment
                if page.page_number == -1 or page.page_number == "-1":
                    continue
                for segment in page.segments:
                    segment_counter = self.get_segment_annotation_counter(segment)
                    counter = counter.add_counter(segment_counter)
        return counter
    
    def update_statistics(self):
        with self.statistics_thread_datalock:
            if self.statistics_thread == None:
                self.statistics_thread = threading.Thread(
                    target=self.update_function)
                self.statistics_thread.start()
                return "Thread is None, starting statistics thread"
            elif not self.statistics_thread.is_alive():
                self.statistics_thread = threading.Thread(
                    target=self.update_function)
                self.statistics_thread.start()
                return "Thread is not alive, starting statistics thread"
            else:
                return "Statistics thread is still running!"
    
    def update_function(self):
        print("Updating statistics")
        # init counter object
        counter = Counter(self.annotation_types)
        
        collections = ["partial", "complete", "total_annotation", "2nd_annotation", "3rd_annotation", "ml_out"]
        for collection in collections:
            t = time.time()
            counter = self.update_counter(counter, collection)
            print("(Statistics) Finished folder", collection, time.time() -t)
        
        # ensure that the object is not read during modification
        with self.statistics_object_datalock:
            self.statistics_object = self.__create_statistics(counter)

class Counter:
    
    def __init__(self, annotation_types, counts=None):
        # init counts object
        self.counts = {}
        self.annotation_types = annotation_types
        if counts == None:
            for key, value in self.annotation_types.items():
                self.counts[key] = {}
                for label in value["labels"]:
                    self.counts[key][label] = 0
        else:
            for key, value in self.annotation_types.items():
                self.counts[key] = {}
                for label in value["labels"]:
                    self.counts[key][label] = counts[key][label]
    
    def add_counter(self, other):
        new_counter = Counter(self.annotation_types, self.counts)
        for key, value in self.annotation_types.items():
            for label in value["labels"]:
                new_counter.counts[key][label] += other.counts[key][label]
        return new_counter
    
    def count(self, annotation_type, label, amount=1):
        if label != None:
            self.counts[annotation_type][label] += amount
    