import json
import os
import urllib
from urllib import request, parse

# CONFIG FILE

def load_config_file():
    with open("/app/config.json", "r", encoding="utf-8") as fp:
        config = json.load(fp)
    return config

# DOCUMENT ANNOTATION STATUS

def get_human_annotated_types(document):
    # returns a list of types for which there exist human annotations
    types = set()
    for annotation_type in document.annotated_by:
        if len(document.annotated_by[annotation_type]) > 0:
            types.add(annotation_type)
    return types

def annotation_is_complete(document, necessary_types):
    # requires that all necessary_types are annotated
    annotated_types = get_human_annotated_types(document)
    for t in necessary_types:
        if t not in annotated_types:
            return False
    return True

def annotation_is_partial(document):
    # requires that at least type 'document_contains' is annotated
    annotated_types = get_human_annotated_types(document)
    if 'document_contains' in annotated_types:
        return True
    else:
        return False

def get_annotation_completion(document, necessary_types):
    if annotation_is_complete(document, necessary_types):
        return "complete"
    elif annotation_is_partial(document):
        return "partial"
    elif document.initial_importance == 1:
        return "priority"
    else:
        return "other"

def get_annotators(document):
    # returns all annotators who annotated block 2 questions
    block_2_set = set(["segment_label", "segment_type", "segment_boundary"])
    annotated_by = document.annotated_by
    annotator_set = set()
    for annotation_type in block_2_set:
        if annotation_type in annotated_by:
            annotators = annotated_by[annotation_type]
            for annotator in annotators:
                print('#'*80)
                print(annotator)
                if type(annotator) == dict:
                    annotator = annotator['name']
                annotator_set.add(annotator)
    return annotator_set
    

# GET / POST REQUESTS

def post_json(data_in, url):
    data = json.dumps(data_in).encode('utf-8')
    req =  urllib.request.Request(url, data=data, headers={'content-type': 'application/json'}) # this will make the method "POST"
    resp = urllib.request.urlopen(req)
    resp_text = resp.read().decode('utf-8')
    return resp_text

def get_json(url):
    req =  urllib.request.Request(url, headers={})
    resp = urllib.request.urlopen(req)
    resp_text = resp.read().decode('utf-8')
    return resp_text

# ANNOTATION DATA ACCESS

def get_annotation_label(annotation, labels):
    vector = annotation.annotation_vector
    if annotation.is_other == "":
        label = get_label_from_vector(vector, labels)
        return label
    else:
        return None

def get_human_annotation(annotation_group, labels):
    # Returns the label of the first human annotation in the group
    ann_type = annotation_group.annotation_type
    for annotation in annotation_group.annotations:
        if is_human_annotator(annotation.annotator_type):
            return get_annotation_label(annotation, labels)
    return None

def is_human_annotator(annotator_type):
    return annotator_type == "human" or annotator_type == "expert" or annotator_type == "non-expert"

def get_label_from_vector(vector, labels):
    # Returns the label with the highest probability score (argmax)
    return labels[max(range(len(vector)), key=lambda i:vector[i] )]
      


# DATA CLASSES

class JSONSerializable():
    # Interface for:
    # - converting an object into valid input for json.dump
    # - converting the output of json.load into an object
    
    def __to_dict__(self):
        # default implementation
        object_dict = {}
        for key, value in self.__dict__.items():
            object_dict[key] = JSONSerializable.serialize(value)
        return object_dict
    
    @staticmethod
    def serialize(o):
        if JSONSerializable.is_json_base(o):
            return o
        if JSONSerializable.is_json_list(o):
            return [JSONSerializable.serialize(value) for value in o]
        if isinstance(o, dict):
            return {key: JSONSerializable.serialize(value) for key, value in o.items()}
        if isinstance(o, JSONSerializable):
            return o.__to_dict__()
        else:
            raise TypeError
    
    @staticmethod
    def deserialize(json_dictionary):
        raise NotImplementedError
    
    @staticmethod
    def is_json_base(o):
        json_defaults = [str, int, float, bool, type(None)]
        return any([isinstance(o, t) for t in json_defaults])
    
    @staticmethod
    def is_json_list(o):
        json_defaults = [list, tuple]
        return any([isinstance(o, t) for t in json_defaults])

class Document(JSONSerializable):
    
    def __init__(self, document_id=-1, annotation_types={}, number_of_pages=0):
        self.document_id = document_id
        self.importance_score = -1
        self.document_label = AnnotationGroup(0, "document_label")
        self.document_type = AnnotationGroup(1, "document_type")
        self.document_contains = AnnotationGroup(2, "document_contains")
        self.annotated_by = {ann_type:[] for ann_type in annotation_types.keys()}
        self.origin = None
        self.initial_importance = -1
        self.annotation_completion = None
        self.target_collection = None
        self.pages = []
        
        for page_number in range(number_of_pages):
            self.pages.append(Page(page_number=page_number))
    
    @staticmethod
    def deserialize(json_dictionary):
        document = Document()
        document.__dict__ = json_dictionary
        document.document_label = AnnotationGroup.deserialize(document.document_label)
        document.document_type = AnnotationGroup.deserialize(document.document_type)
        document.document_contains = AnnotationGroup.deserialize(document.document_contains)
        document.pages = [Page.deserialize(page) for page in document.pages]

        return document
    
    def annotations(self):
        for page in self.pages:
            for annotation in page.annotations():
                yield annotation
                

class Page(JSONSerializable):
    
    def __init__(self, page_number=-1):
        self.page_number = page_number
        self.segments = []
    
    @staticmethod
    def deserialize(json_dictionary):
        page = Page()
        page.__dict__ = json_dictionary
        page.segments = [Segment.deserialize(segment) for segment in page.segments]
        return page
    
    def annotations(self):
        for segment in self.segments:
            for annotation in segment.annotations():
                yield annotation

class Segment(JSONSerializable):
    
    def __init__(self, segment_id=-1):
        self.segment_id = segment_id
        self.child_segments = []
        self.annotation_groups = {}
    
    @staticmethod
    def deserialize(json_dictionary):
        segment = Segment()
        segment.__dict__ = json_dictionary
        segment.child_segments = [Segment.deserialize(segment) for segment in segment.child_segments]
        for annotated_type, group in segment.annotation_groups.items():
            segment.annotation_groups[annotated_type] = AnnotationGroup.deserialize(group)
        return segment
    
    def annotations(self):
        for annotated_type, group in self.annotation_groups.items():
            for annotation in group.annotations:
                yield annotation 
        for segment in self.child_segments:
            for annotation in segment.annotations():
                yield annotation

class AnnotationGroup(JSONSerializable):
    
    def __init__(self, group_id=-1, annotation_type=None):
        self.group_id = group_id
        self.importance_score = -1
        self.annotation_type = annotation_type
        self.annotations = []
    
    @staticmethod
    def deserialize(json_dictionary):
        group = AnnotationGroup()
        group.__dict__ = json_dictionary
        tmp = []
        for annotation in group.annotations:
            ann_obj = Annotation.deserialize(annotation)
            ann_obj.annotation_type = group.annotation_type
            tmp.append(ann_obj)
        group.annotations = tmp
        return group


class Annotation(JSONSerializable):
    
    def __init__(self, annotation_id=-1, annotator_id=None, annotator_type=None, annotation_vector=None, annotation_type=None, is_other="", annotation_version=None):
        self.annotation_id = annotation_id
        self.annotator_id = annotator_id
        self.annotator_type = annotator_type
        self.annotation_vector = annotation_vector
        self.annotation_type = annotation_type
        self.is_other = is_other
        self.annotation_version = annotation_version
    
    @staticmethod
    def deserialize(json_dictionary):
        annotation = Annotation()
        annotation.__dict__ = json_dictionary
        return annotation
    
    @staticmethod
    def label2vector(label_text="", annotation_type="", annotation_types={}):
        # do not use if list of labels is large
        # or label probabilities are provided by the model
        # returns None if label_text not found
        if label_text not in annotation_types[annotation_type]["labels"]:
            return None
        labels = annotation_types[annotation_type]["labels"]
        idx = labels.index(label_text)
        return [1.0 if i == idx else 0.0 for i in range(len(labels))]


