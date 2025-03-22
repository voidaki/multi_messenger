from ligo.gracedb.rest import GraceDb
import numpy as np


client = GraceDb()
superevent_iterator = client.superevents("gpstime: 1123858817 .. 1443884418")  # gpstime: 1123858817 .. 1443884418 (O1 to end of 04) about 4000 samples
superevent_fars = [superevent["far"] for superevent in superevent_iterator]
fars = np.array(superevent_fars)
print(fars)
np.savetxt("false_alarm_list.csv", fars, delimiter=",")