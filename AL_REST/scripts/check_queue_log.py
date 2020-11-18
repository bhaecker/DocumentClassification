import datetime
import time
import calendar

# datetime object
now = datetime.datetime.now()

epoch = datetime.datetime.utcfromtimestamp(0)

def unix_time_millis(dt):
    return (dt - epoch).total_seconds() #* 1000.0

date = datetime.datetime(2020, 10, 29, 14, 22, 0, 0)

time_of_change = float(unix_time_millis(date))


path_to_queue_log = r"C:\Users\anna_\Documents\Active_Learning\logs-WS-Tag1\queue_log.txt"

id_list = []
with open(path_to_queue_log) as fp:
    line = fp.readline()
    while line:
        s = line.split("]")[0]
        s = s.replace("[time: ","")
        #print(s)
        time = float(s)

        print(time)
        print(time_of_change)
        if time > time_of_change:

            if "next_document" in line and "document_id" in line:
                s = line.split("document_id': '")
                #print(s[1])
                s = s[1].split("',")
                id_list.append(s[0])

        line = fp.readline()   

double = 0
not_double = 0
for idx, viewed in enumerate(id_list):
    if ([i for i, e in enumerate(id_list) if e == viewed] == [idx]):
        double += 1
    else:
        not_double +=1

print("double " + str(double))
print("not_double " + str(not_double))

