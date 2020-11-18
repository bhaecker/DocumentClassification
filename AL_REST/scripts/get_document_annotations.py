import json
import os
import datetime
import dateutil.parser
import argparse
from pprint import pprint

def get_page_annotations(page, annotator_id):
    segs = []
    for segment in page['segments']:
        seg_data = get_segment_annotations(segment, annotator_id)
        if seg_data == None:
            continue
        segs.append(seg_data)
    return segs

def get_segment_annotations(segment, annotator_id):
    groups_dict = segment['annotation_groups']
    segment_type = get_annotation_group_label(groups_dict['segment_type'],
                                              annotation_types['segment_type']['labels'], annotator_id)
    segment_label = get_annotation_group_label(
        groups_dict['segment_label'], 
        annotation_types['segment_label']['labels'], annotator_id)
    """
    if (segment_type==None) != (segment_label==None):
        if segment_type == None:
            print("### TYPE NONE")
        if segment_label == None:
            print("### LABEL NONE")
        pprint(segment['annotation_groups'])
    if segment_label == None or segment_type == None:
        return None
    if segment_label not in annotation_types['segment_label']['labels']:
        if segment_label not in custom_labels:
            custom_labels[segment_label] = 1
        else:
            custom_labels[segment_label] += 1
        segment_label = None"""
    # "xmin","ymin","xmax","ymax"
    seg_data = get_bounding_box(groups_dict['segment_boundary'], 
                                annotation_types['segment_boundary']['labels'], annotator_id)
    if seg_data == None:
        return None
    seg_data['segment_label'] = segment_label
    seg_data['segment_type'] = segment_type
    seg_data['annotator_id'] = annotator_id
    return seg_data

def get_bounding_box(annotation_group, labels, annotator_id):
    annotation = None
    for ann in annotation_group['annotations']:
        if ann['annotator_id'] == annotator_id:
            annotation = ann
    if annotation == None:
        return None
    return {label:value for label, value in zip(labels,annotation['annotation_vector'])}

def get_annotation_group_label(annotation_group, labels, annotator_id):
    if len(annotation_group['annotations']) == 0:
        return None
    # there should be exactly one label per bounding box right now
    annotation = None
    for ann in annotation_group['annotations']:
        if ann['annotator_id'] == annotator_id:
            annotation = ann
    if annotation == None:
        return None
    label = get_annotation_label(annotation, labels)
    return label

def get_annotation_label(annotation, labels):
    vector = annotation['annotation_vector']
    if annotation['is_other'] == "":
        label = get_label_from_vector(vector, labels)
    else:
        label = annotation['is_other']
        if label == None:
            label = 'is_other' 
    return label

def get_label_from_vector(vector, labels):
    return labels[max(range(len(vector)), key=lambda i:vector[i] )]

def get_annotators(document):
    # returns the annotators of a document sorted by annotation time
    # if an annotator appears multiple times in a document, only adds the first timestamp
    annotators = set()
    for ann_type, ann_list in document['annotated_by'].items():
        annotators = annotators | set([(d['name'], d['timestamp']) for d in ann_list])
    annotators_list = [{'document_id':document['document_id'], 
                        'annotator_id':name, 
                        'timestamp':dateutil.parser.parse(timestamp)} for name, timestamp in annotators]
    tmp_list = sorted(annotators_list, key=lambda d:d['timestamp'])
    annotators_list = []
    annotators = set()
    for annotator in tmp_list:
        if annotator['annotator_id'] not in annotators:
            annotators.add(annotator['annotator_id'])
            annotators_list.append(annotator)
    return annotators_list

def get_document_annotations(document, annotator_id):
    doc_segs = []
    document_label = get_annotation_group_label(document['document_label'],
                                                annotation_types['document_label']['labels'], annotator_id)
    document_type = get_annotation_group_label(document['document_type'],
                                               annotation_types['document_type']['labels'], annotator_id)
    for page in document['pages']:
        if page['page_number'] == -1:
            continue
        doc_segs.extend(get_page_annotations(page, annotator_id))
    return doc_segs, {'document_label':document_label, 
                      'document_type':document_type, 'annotator_id': annotator_id}

def print_annotations(document):
    for annotator in annotators:
        doc_segs, doc_data = get_document_annotations(document, annotator['annotator_id'])
        print("\nAnnotator")
        pprint(annotator)
        print("\nDocument Label & Type")
        pprint(doc_data)
        print("\nSegment Label & Type")
        pprint(doc_segs)

def print_total_annotations(document):
    for idx, annotator in enumerate(annotators):
        if idx == 0:
            continue
        doc_segs, doc_data = get_document_annotations(document, annotator['annotator_id'])
        print("\nAnnotator")
        pprint(annotator)
        print("\nDocument Label & Type")
        pprint(doc_data)
        print("\nSegment Label & Type")
        pprint(doc_segs)
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
    """Prints the annotations of a document.""")
    parser.add_argument("document", help="path to the document")
    parser.add_argument("annotation_types", help="path to the annotation_types file")
    #parser.add_argument("-t", "--total_annotation", type=bool, default=False, help="whether the document is a total_annotation document")
    args = parser.parse_args()
    
    with open(args.annotation_types, 'r') as fp:
        annotation_types = json.load(fp)
    
    with open(args.document, 'r') as fp:
        document = json.load(fp)
    
    annotators = get_annotators(document)
    
    print_annotations(document)
    """
    if args.total_annotation == True:
        print_total_annotations(document)
    else:
        print_annotations(document)
    """
        
        