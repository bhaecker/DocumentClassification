from collections import Counter

#path = r"C:\Users\anna_\Documents\Active_Learning\logs-WS-Tag3\log.txt"
path = r"C:\Users\anna_\Documents\Active_Learning\logs-WS-Tag3\log-2\log.txt"

# double?
#ad7fc1a13adf510377182776c680eac9

next_docs = {}
returned_docs = {}

all_returned_docs = []
all_next_docs = []


next_doc = ""
returned_doc = ""

saved_docs = []

with open(path) as fp:
    line = fp.readline()
    while line:

        # lets follow one doc
        if "4f9eca888141f0de0d7f83b526e8e6da" in line: # saved 2x
            print(line)
        if "ERROR<class 'pymongo.errors.CursorNotFound'>" in line:
            print(line)

        if "next_document: {" in line:
            line = line.split("document_id': '")[1]
            next_doc = line.split("'")[0]
            #print(next_doc)
            all_next_docs.append(next_doc)

        if "al_complete: return document" in line:
            returned_doc = line.split("al_complete: return document ")[1]
            returned_doc = returned_doc.strip()
            all_returned_docs.append(returned_doc)

        if "annotator_id: " in line:
            annotator_id = line.split("annotator_id: ")[1].strip()

            returned_docs[returned_doc] = annotator_id
            #next_docs[next_doc] = annotator_id

            next_doc = ""
            returned_doc = ""
            annotator_id = ""
        
        if "add/update document" in line:
            stored_id = line.split("add/update document ")[1]
            stored_id = stored_id.strip()
            saved_docs.append(stored_id)
            #print(stored_id)


        #if "ERROR<class 'pymongo.errors.CursorNotFound'>" in line:
        #    break # break on queue error
        # stop when view queue starts (shows get next that are not being annotated)
        if "QueryQueue show documents that match query: {'$and': [{'has_finetuned': True}]}" in line:
            break

        line = fp.readline()


# issue ---> queue error
# was ist mit blocked ids???


# for every next there should be one save
# every next should only be one

#counter = 0
#for idx, next_item in enumerate(next_docs):
#    for idx2, next_item2 in enumerate(next_docs):
#        #if next_item == next_item2 and idx!=idx2:
#        if next_item == next_item2:
#            counter = counter+1

#print("counted doubles in next", counter)
#print("next len ",len(next_docs))


# fuck returned die queue immer wieder dasselbe????
# also übersprungene wären ja ok ....
# warum so wenige saves???

counter = 0
for idx, next_item in enumerate(all_returned_docs):
    for idx2, next_item2 in enumerate(all_returned_docs):
        #if next_item == next_item2 and idx!=idx2:
        if next_item == next_item2:
            counter = counter+1
            #print(next_item)

#for idx, viewed in enumerate(viewed_docs):
#    assert([i for i, e in enumerate(viewed_docs) if e == viewed] == [idx])

print("count all returned ", counter)
print(len(all_returned_docs))
print(counter/len(all_returned_docs))

# every doc is shown like 5 times
# saved ones too? after being saved??
# to different people?

counter = 0
for idx, next_item in enumerate(all_next_docs):
    for idx2, next_item2 in enumerate(all_next_docs):
        #if next_item == next_item2 and idx!=idx2:
        if next_item == next_item2:
            counter = counter+1
            #print(next_item)

print("count all next ", counter)
print(len(all_next_docs))

print(counter/len(all_next_docs))


counter = 0
for idx, next_item in enumerate(returned_docs):
    for idx2, next_item2 in enumerate(returned_docs):
        if next_item == next_item2 and idx!=idx2:
        #if next_item == next_item2:
            counter = counter+1

print("counted doubles in returned", counter)
print("returned len ",len(returned_docs))

# only in next_document??
for idx, next_item in enumerate(next_docs):
    if next_item == "09bfcf3e8cffde60947b97e0550ffdaa":
        print("OK - next")

for idx, next_item in enumerate(returned_docs):
    if next_item == "09bfcf3e8cffde60947b97e0550ffdaa":
        print("OK - returned")

# 09bfcf3e8cffde60947b97e0550ffdaa only once 


count_saved = 0
#for saved_item in saved_docs:
for item in saved_docs:  
    count_saved += 1

print("count saved ", count_saved)

not_saved = {}
saved = {}
for idx, next_item in enumerate(returned_docs):
    found = False
    if not next_item in saved_docs:
        #not_saved.append(next_item)
        annotator = returned_docs[next_item]
        not_saved[next_item] = annotator

    else:
        #saved.append(next_item)
        annotator = returned_docs[next_item]
        saved[next_item] = annotator

print("len not saved ", len(not_saved))
print("len saved ", len(saved))
#print(not_saved)


# repeat in saved?
counter = 0
for idx, next_item in enumerate(saved_docs):
    for idx2, next_item2 in enumerate(saved_docs):
        if next_item == next_item2 and idx!=idx2:
            counter = counter+1

print("saved repeat ", counter)

# überspringen möglich ...
# mehr saved - man kann mehrmals dasselbe speichern (?)


print("total complete docs labeled not is other")
print( sum([37, 68, 12, 9, 8, 12, 11, 0, 5, 0, 0, 10, 0, 3, 22, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 10, 0]))

# saved docs per person
print(Counter(saved.values()))
#print(Counter(not_saved.values()))
