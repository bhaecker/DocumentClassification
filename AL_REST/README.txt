0.
install Docker
with andaconda create and activate an environment (with python 3.6) 
and install all the necessary dependencies (see requirements.txt)

1. run inside docker folder in a cmd window: 
docker build --tag="rest_al:latest" .
docker-compose -f docker-compose.yml up
(see docker_script)

these calls use docker-compose.yaml and Dockerfile
if you add a new dependency (import xyz): add [xyz] to requirements.txt

(if you change the code (except in testing.py) you have to do a new build and compose
for changes to take effect)

2. for testing
in docker folder run:
python -m efficient_annotation.testing

add new test cases in testing.py:

test_function_name: 
    # function that tests something
    # use assert to check result

t_name = [
    ("description", test_function_name, [parameter1, parameter2], {}), 
    ("description 2", test_function_name2, [], {})
    ]

tests = [
...
    t15, 
    t_name
]

---------------------------------------------------------------------

communication with outside components (machine learning models, GUI)
through rest calls (REST_AL.py)
 
/add_document - inserts/updates a document in the database

/init_next_most_important/<path:queue_type> - starts a queue
queues (queuing/queues.py) are handled by queue manager (queuing/managers.py)
a queue retrieves the sorted documents from datastores/mongodb.py
when starting a new queue, that has a score_calculator, all documents 
receive an importance_score (that is calculated with score_calculator) 

/next_most_important/<path:annotator_id> - gets the next document that this annotator has not seen yet.
exception: annotator can see document once for partial annotation and once for detailed annotation.
(sorted by importance score, and number of annotators)
when a document retrieves an annotation its target_collection is set to target (which is set in generator in queue)
(but this is done by GUI component)



ML output: json with annotated_by and annotation_vector per segment; target_collection = "ml_output"
GUI input: json (calls next_most_important and add_document)


