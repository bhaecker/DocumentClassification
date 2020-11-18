import datetime
import time
import calendar

# datetime object
now = datetime.datetime.now()

epoch = datetime.datetime.utcfromtimestamp(0)

def unix_time_millis(dt):
    return (dt - epoch).total_seconds() #* 1000.0

date = datetime.datetime(2020, 10, 18, 22, 24, 5, 95)

print(unix_time_millis(date))