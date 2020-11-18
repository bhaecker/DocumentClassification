import mongodbshell

# connect to db
mongodb_uri = 'mongodb://localhost:27017' #"mongodb://document-database:27017/"
client = mongodbshell.MongoDB('database', 'documents', mongodb_uri)
collection = client.database.documents

################################################################################
# get all documents where document label was annotated by 
# annotator_id == "FR"
# and document label is Angebot

#annotator_id = "FR"
#vector_angebot = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#
#query = {
#        "$and":[{
#        "document_label.annotations.annotator_id": annotator_id,
#        "document_label.annotations.annotation_vector": vector_angebot
#        }]
#        }
#include_values = { "_id": 1}
#cursor = collection.find(query, include_values)
#
#for item in cursor:
#    print(item["_id"])

################################################################################
# get all ids with some segment_label
is_other_segment_label = "UZ"

query = {
        "pages.segments.annotation_groups.segment_label.annotations.is_other": is_other_segment_label
        }

include_values = { "_id": 1, "annotated_by.annotators_complete":1}
cursor = collection.find(query, include_values)

counter = 0 
for item in cursor:
    print(item)
    counter += 1

print("total: ", counter)
