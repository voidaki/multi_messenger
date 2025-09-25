import pprint
from gcn_kafka import Consumer
import xml.etree.ElementTree as ET
import xmltodict
import datetime

# Convert a raw XML string into an ElementTree XML object
def parse_voevent_alert_to_xml_root(message_value):
    return ET.fromstring(message_value)

def parse_voevent_alert_to_dict(message_value):
    return xmltodict.parse(message_value)

# Connect as a consumer (client "heimdall")
# Warning: don't share the client secret with others.
consumer = Consumer(client_id='29rsbk1br7u12ut8bm8bhafnau',
                    client_secret='1jil45es9lfs5s6k2j6mkab089aqfhft5v7hebv5qvbi6g8393df')

consumer.subscribe([
                    'gcn.classic.voevent.FERMI_GBM_ALERT',
                    'gcn.classic.voevent.FERMI_GBM_FIN_POS',
                    'gcn.classic.voevent.FERMI_GBM_FLT_POS',
                    'gcn.classic.voevent.FERMI_GBM_GND_POS',
                    'gcn.classic.voevent.FERMI_GBM_POS_TEST',
                    'gcn.classic.voevent.FERMI_GBM_SUBTHRESH',
                    'gcn.classic.voevent.FERMI_LAT_MONITOR',
                    'gcn.classic.voevent.FERMI_LAT_OFFLINE',
                    'gcn.classic.voevent.FERMI_LAT_POS_TEST',
                    'gcn.classic.voevent.FERMI_POINTDIR',
                    'gcn.classic.voevent.ICECUBE_ASTROTRACK_BRONZE',
                    'gcn.classic.voevent.ICECUBE_ASTROTRACK_GOLD',
                    'gcn.classic.voevent.ICECUBE_CASCADE'
                    # 'gcn.classic.voevent.LVC_COUNTERPART',
                    # 'gcn.classic.voevent.LVC_EARLY_WARNING',
                    # 'gcn.classic.voevent.LVC_INITIAL',
                    # 'gcn.classic.voevent.LVC_PRELIMINARY',
                    # 'gcn.classic.voevent.LVC_RETRACTION',
                    # 'gcn.classic.voevent.LVC_UPDATE'
                    ])
while True:
    for message in consumer.consume(timeout=1):
        if message.error():
            print(message.error())
            continue
        # Print the topic and message ID
        print(f'topic={message.topic()}, offset={message.offset()}')
        print("type of the message: ", type(message))
        
        value = message.value()
        print("type of the value:", type(value))
        xml_root = parse_voevent_alert_to_xml_root(value)
        dictionary = parse_voevent_alert_to_dict(value)
        print(value)
        print("\n")
        print(xml_root)
        print("\n")
        print(dictionary)
        print("\n")
        pprint.pprint(dictionary)

        with open(f"./gcn_notices/{datetime.datetime.now()}.xml", "wb") as f:
            f.write(value)

