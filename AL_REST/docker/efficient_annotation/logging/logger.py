import time
import os
import sys
from datetime import datetime, timezone
from efficient_annotation.common import load_config_file



class Logger:
    def __init__(self):
        self.path_to_log_file = os.path.join(self.get_data_path(),"log.txt")
        print("#### write log to " + self.path_to_log_file + " ####", flush = True)

    def log(self, output, filename = None, mode = None):

        if mode == None:
            mode = "a"

        if filename != None:
            path = os.path.join(self.get_data_path(), str(filename))
        else: 
            path = self.path_to_log_file

        # TODO: fix: added 1 hour to make time correct 
        message = ("[" + str(time.time())
        + " | " 
        + str(time.localtime().tm_year) + "/"
        + str(time.localtime().tm_mon) + "/"
        + str(time.localtime().tm_mday) + " "
        + str(time.localtime().tm_hour+1)+ ":" 
        + str(time.localtime().tm_min) +":"
        + str(time.localtime().tm_sec) 
        + "] " 
        + str(output))
        print(message, flush = True)
        # open log file
        f = open(path, mode)
        # write output
        f.write(message+"\n")
        # close log file
        f.close()


    # store without time stamp
    def log_output_only(self, output, filename = None, mode = None):
        if mode == None:
            mode = "a"

        if filename != None:
            path = os.path.join(self.get_data_path(), str(filename))
        else: 
            path = self.path_to_log_file

        #message = "[time: " + str(time.time())+ "] " + str(output)

        #print(message, flush = True)
        # open log file
        f = open(path, mode)
        # write output
        f.write(str(output))
        # close log file
        f.close()


    def get_path_to_logged_file(self, filename):
        return os.path.join(self.get_data_path(), str(filename))

    def empty_log(self, filename):
        if os.path.isfile(os.path.join(self.get_data_path(), str(filename))):
            path = os.path.join(self.get_data_path(), str(filename))
            f = open(path, "w")
            f.close()
        else:
            print("WARNING: empty_log did not find file " + str(os.path.join(self.get_data_path(), str(filename))))


    def get_data_path(self):
        config = load_config_file()
        return config["data"]["path"] 