# PATH TO LOG FILE #

#path_to_log = r"C:\Users\anna_\Documents\Active_Learning\results echtfall test 5000\ap2\log.txt"
path_to_log = r"C:\Users\anna_\Documents\Active_Learning\log-19-10-20\krex-shared\ap2\log.txt"

al_random_start = -1
with open(path_to_log) as fp:
    line = fp.readline()
    while line:
        if "ERROR" in line:
            print(line)
        if "WARNING" in line and not "no document with document_id" in line:
            print(line)

        line = fp.readline()            
