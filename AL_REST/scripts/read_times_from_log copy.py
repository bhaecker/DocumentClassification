# PATH TO LOG FILE #
#path_to_log = r"C:\Users\anna_\Downloads\logs-29-09-20\log.txt"
path_to_log = r"C:\Users\anna_\Downloads\log-07-10-2020\run2\log.txt"


def print_elapsed_time(start_time, end_time, num_docs, name):
    print("{1:10.2f} sec {2:10.2f} min  {0} ({3} docs)".format(name, end_time-start_time, (end_time-start_time)/60, num_docs))

def print_elapsed_time_per_doc(start_time, end_time, num_docs, name):
    per_doc = (end_time-start_time)/num_docs
    print("{1:10.2f} sec {2:10.2f} min  {0} per doc".format(name, per_doc, per_doc/60))
  

def print_estimated_elapsed_time_with_x_docs(start_time, end_time, num_docs, name, x):
    per_doc = (end_time-start_time)/num_docs
    print("{1:10.2f} sec {2:10.2f} min {3:10.2f} h {4:10.2f} d  {0} ({5} docs)".format(name, per_doc*x, (per_doc*x)/60, (per_doc*x)/(60*60), (per_doc*x)/(60*60*24), x))


al_random_start = -1
with open(path_to_log) as fp:
    line = fp.readline()
    while line:

        if "number of documents in target_collections has_complete_human_annotation and has_partial_human_annotation before removing documents" in line:
            total_docs = line.split("] ")[1]
            total_docs = total_docs.replace("number of documents in target_collections has_complete_human_annotation and has_partial_human_annotation before removing documents:","")
            total_docs = int(total_docs.strip())
            print("total number of documents with human annotation:", total_docs)

        if "number of documents after removing documents with zero-only human annotation and documents with" in line:
            filtered_docs = line.split("] ")[1]
            print(filtered_docs.replace("\n", ""))
            filtered_docs = filtered_docs.replace("number of documents after removing documents with zero-only human annotation and documents with","")
            filtered_docs = filtered_docs.split(":")[1]
            filtered_docs = int(filtered_docs.strip())

        # split
        if "split dataset with test_split" in line:
            time = line.split("] ")[0].replace("[time: ","")
            split_start = float(time)
            total_evaluation_start = float(time)

        if "first dataset split finished" in line:
            time = line.split("] ")[0].replace("[time: ","")
            split_end = float(time)
            #print("time elapsed for dataset split: ", split_end-split_start, "s")   

        # base prediction
        if "start_base_predictions" in line:
            time = line.split("] ")[0].replace("[time: ","")
            base_start = float(time)

        if "Done: base_predictions" in line:
            time = line.split("] ")[0].replace("[time: ","")
            base_end = float(time)
            #print("time elapsed for base_predictions: ", base_end-base_start, "s")


        # initial finetuning
        if "start_finetuning_training" in line and not "Done: start_finetuning_training" in line:
            time = line.split("] ")[0].replace("[time: ","")
            initial_ft_start = float(time)

        if "Done: start_finetuning_training" in line:
            time = line.split("] ")[0].replace("[time: ","")
            initial_ft_end = float(time)
            #print("time elapsed for initial finetuning: ", initial_ft_end-initial_ft_start, "s")


        # active learning: efficient annotation 
        if "start_active_learning with type efficient_annotation" in line:
            time = line.split("] ")[0].replace("[time: ","")
            al_efficient_start = float(time)

        if "start_active_learning with type random" in line:
            time = line.split("] ")[0].replace("[time: ","")
            al_efficient_end = float(time)
            #print("time elapsed for active learning: efficient annotation: ", al_efficient_end-al_efficient_start, "s")


        # active learning: random
        if "start_active_learning with type random" in line:
            time = line.split("] ")[0].replace("[time: ","")
            al_random_start = float(time)

        if "[time: " in line:
            last_line_with_time = line

        line = fp.readline()            

    # at end of file: active learning: random
    time = last_line_with_time.split("] ")[0].replace("[time: ","")
    al_random_end = float(time)
    total_evaluation_end = float(time)
    #print("time elapsed for active learning: random: ", al_random_end-al_random_start, "s")

    #print("total time elapsed for evaluation: ", total_evaluation_end-total_evaluation_start, "s", " = ", (total_evaluation_end-total_evaluation_start)/60, "min")
    #print("total time per doc: ", (total_evaluation_end-total_evaluation_start)/filtered_docs, "s", " = ", ((total_evaluation_end-total_evaluation_start)/filtered_docs)/60, "min")



    print("\ntime elapsed:\n")
    print_elapsed_time(split_start, split_end, filtered_docs, "split")
    print_elapsed_time(base_start, base_end, filtered_docs, "base prediction")
    print_elapsed_time(initial_ft_start, initial_ft_end, filtered_docs, "initial finetuning")
    print_elapsed_time(al_efficient_start, al_efficient_end, filtered_docs, "active learning: efficient_annotation")
    print_elapsed_time(al_random_start, al_random_end, filtered_docs, "active learning: random")
    print_elapsed_time(total_evaluation_start, total_evaluation_end, filtered_docs, "evaluation total")

    print("\ntime elapsed per doc:\n")

    print_elapsed_time_per_doc(split_start, split_end, filtered_docs,"split")
    print_elapsed_time_per_doc(base_start, base_end, filtered_docs, "base prediction")
    print_elapsed_time_per_doc(initial_ft_start, initial_ft_end, filtered_docs, "initial finetuning")
    print_elapsed_time_per_doc(al_efficient_start, al_efficient_end, filtered_docs, "active learning: efficient_annotation")
    print_elapsed_time_per_doc(al_random_start, al_random_end, filtered_docs, "active learning: random")
    print_elapsed_time_per_doc(total_evaluation_start, total_evaluation_end, filtered_docs, "evaluation total")


    x = 7000
    print(("\nestimated time with {0} docs:\n").format(x))

    print_estimated_elapsed_time_with_x_docs(split_start, split_end, filtered_docs,"split",x)
    print_estimated_elapsed_time_with_x_docs(base_start, base_end, filtered_docs, "base prediction",x)
    print_estimated_elapsed_time_with_x_docs(initial_ft_start, initial_ft_end, filtered_docs, "initial finetuning",x)
    print_estimated_elapsed_time_with_x_docs(al_efficient_start, al_efficient_end, filtered_docs, "active learning: efficient_annotation",x)
    print_estimated_elapsed_time_with_x_docs(al_random_start, al_random_end, filtered_docs, "active learning: random",x)
    print_estimated_elapsed_time_with_x_docs(total_evaluation_start, total_evaluation_end, filtered_docs, "evaluation total",x)
