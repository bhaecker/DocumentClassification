# PATH TO LOG FILE #
path_to_log = r"C:\Users\anna_\Downloads\logs-30-09-20\log.txt"


def print_elapsed_time(start_time, end_time, num_docs, name):
    print("{1:10.2f} sec {2:10.2f} min  {0} ({3} docs)".format(name, end_time-start_time, (end_time-start_time)/60, num_docs))

def print_elapsed_time_per_doc(start_time, end_time, num_docs, name):
    per_doc = (end_time-start_time)/num_docs
    print("{1:10.2f} sec {2:10.2f} min  {0} per doc".format(name, per_doc, per_doc/60))
  

def print_estimated_elapsed_time_with_x_docs(start_time, end_time, num_docs, name, x):
    per_doc = (end_time-start_time)/num_docs
    print("{1:10.2f} sec {2:10.2f} min {3:10.2f} h {4:10.2f} d  {0} ({5} docs)".format(name, per_doc*x, (per_doc*x)/60, (per_doc*x)/(60*60), (per_doc*x)/(60*60*24), x))


with open(path_to_log) as fp:
    line = fp.readline()
    while line:

        # start
        if "start_prediction on DOCUMENT_OVERVIEW_CSV" in line:
            time = line.split("] ")[0].replace("[time: ","")
            first_base_predict_start = float(time)

        if "return from call predict_batch_with_base_model" in line:
            time = line.split("] ")[0].replace("[time: ","")
            first_base_predict_end = float(time)


        line = fp.readline()            



    print("\ntime elapsed:\n")
    filtered_docs = 989 # TODO

    print_elapsed_time(first_base_predict_start, first_base_predict_end, filtered_docs, "first base predict")
    print_elapsed_time_per_doc(first_base_predict_start, first_base_predict_end, filtered_docs, "first base predict")
