
def simple_diversity_score(document, weights):
    # weights needs to be a weight vector of the same length as
    # the document_contains vector with values between 0.0 and 1.0
    sum = 0.0
    # get document_contains annotation
    annotations = document.document_contains.annotations
    if len(annotations) == 0:
        return sum
    annotation = annotations[0]
    for v, w in zip(annotation.annotation_vector, weights):
        sum += v*w
    return sum

def calculate_weights(statistics, annotation_types, important_types):
    # calulates a weights vector from the annotation_types listed in
    # important_types by concatenating their labels
    weights = []
    for ann_type in important_types:
        labels = annotation_types[ann_type]['labels']
        for label in labels:
            total, expected = statistics[ann_type][label]
            #print(ann_type, label, total, expected)
            weight = max((1-total/expected), 0.0)
            weights.append(weight)
    return weights

class ScoreCalculator:
    # The score calculator calculates the importance score of a document
    # out of two parts:
    # 1. based on the document score_data is calculated with the scoring_function
    # 2. based on the statistics of the current annotation session the score_data
    #    is weighted
    
    def scoring_function(self, document, weights = None):
        raise NotImplementedError
    
    def weighted_score(self, score_data, statistics):
        raise NotImplementedError

class DiversityScoreCalculator(ScoreCalculator):
    
    def __init__(self, annotation_types=None, important_types=None):
        self.annotation_types = annotation_types
        self.important_types = important_types
    
    def scoring_function(self, document, weights = None):
        # get document_contains annotation
        annotations = document.document_contains.annotations
        if len(annotations) == 0:
            return [0]*len(self.annotation_types['document_contains']['labels'])
        annotation = annotations[0]

        #print("DiversityScoreCalculator: result of scoring_function " + str(annotation.annotation_vector), flush = True)

        # ADDED
        #if weights != None:
        #    print("return weighted score", flush = True)
        #    return self.weighted_score(annotation.annotation_vector, weights)
        # end: ADDED
        # else:
        # print("weights = None", flush = True)
        return annotation.annotation_vector
    
    def weighted_score(self, score_data, statistics):

        statistics = statistics() # call get_statistics method to get statistics

        # calulates a weights vector from the annotation_types listed in
        # important_types by concatenating their labels
        weights = []
        for ann_type in self.important_types:
            labels = self.annotation_types[ann_type]['labels']
            for label in labels:
                total, expected = statistics[ann_type][label]
                #print(ann_type, label, total, expected)
                weight = max((1-total/expected), 0.0)
                weights.append(weight)

        print("score_data = " + str(score_data))
        print("weight = " + str(weights), flush = True)

        # calulate the weighted_score
        sum = 0.0
        for v, w in zip(score_data, weights):
            sum += v*w

        print("DiversityScoreCalculator: result of weighted_score " + str(sum), flush = True)
        return sum

class InitialScoreCalculator(ScoreCalculator):
    
    def scoring_function(self, document, weights = None):
        return document.initial_importance
    
    def weighted_score(self, score_data, statistics):
        return score_data


class AnnotationVectorScoreCalculator(ScoreCalculator):
    
    def __init__(self, annotation_types=None, important_types=None, dist_scoring_fn=None, w_avg=0.5, w_max=0.5):
        # w_avg: scoring weight for the entropy average of the 
        # w_max: scoring weight for the entropy maximum of the document
        self.annotation_types = annotation_types
        self.important_types = important_types
        self.dist_scoring_fn = dist_scoring_fn
        self.w_avg = w_avg
        self.w_max = w_max
    
    def scoring_function(self, document, weights = None):

        # get document_contains annotation
        scores = []
        for annotation in document.annotations():
            if self.is_important_annotation(annotation):
                #print("is important annotation")
                #print(annotation.annotation_type)
                score = self.calculate_annotation_score(annotation)
                annotation.importance_score = score
                scores.append(score)
            #else:
                #print("not important annotation")
                #print(annotation.annotation_type)
        return self.calculate_document_score(scores)
    
    def calculate_annotation_score(self, annotation):
        # Calculates the annotation score
        vector = annotation.annotation_vector

        #print("calculate for vector:");print(vector)
        #print(self.dist_scoring_fn(vector), flush = True)

        return self.dist_scoring_fn(vector)
    
    def is_important_annotation(self, annotation):
        return annotation.annotation_type in self.important_types
    
    def calculate_document_score(self, scores):
        if len(scores) == 0:
            return 0
        avg = sum(scores)/len(scores)
        maximum = max(scores)

        #print("**** accumulated document score ****")
        #print(avg*self.w_avg + maximum*self.w_max)

        return avg*self.w_avg + maximum*self.w_max
    
    def weighted_score(self, score_data, statistics):
        return score_data