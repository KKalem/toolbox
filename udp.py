# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 19:35:31 2017

@author: https://www.raspberrypi.org/forums/viewtopic.php?f=32&t=39431
"""

import socket
import time
import pickle

import threading
import queue

HOST_IP = "0.0.0.0" # all interfaces
SENDER_PORT = 1501
# 224.0.1.0 thru 224.255.255.255
# (ping 224.0.0.1 for the group mulitcast server list)
MCAST_ADDR = "224.168.2.9"
MCAST_PORT = 1600
TTL=31# valid value are 1-255, <32 is local network

# some time to sleep between bcasts
SLEEP_TIME = 0.5

class Producer:
    def __init__(self, sender_ip=HOST_IP, sender_port=SENDER_PORT, ttl=TTL):
        try:
            self.sock = socket.socket(socket.AF_INET,
                                      socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            self.sock.bind((sender_ip,sender_port))
            self.sock.setsockopt(socket.IPPROTO_IP,
                                 socket.IP_MULTICAST_TTL, ttl)
        except socket.error as e:
            if socket.error == 10048:
                self.__init__(sender_ip,sender_port+1,ttl)

    def send(self, msg="", mcast_addr=MCAST_ADDR, mcast_port=MCAST_PORT,):
        pickled_msg = pickle.dumps(msg)
        self.sock.sendto(pickled_msg, (mcast_addr, mcast_port))

    def host_name(self):
        return socket.gethostname()

class Consumer:
    def __init__(self, client_ip=HOST_IP, mcast_addr=MCAST_ADDR,
                 mcast_port=MCAST_PORT, ttl=TTL, blocking=0):
        self.sock = socket.socket(socket.AF_INET,
                                  socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        self.sock.bind((client_ip, mcast_port))
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                             socket.inet_aton(mcast_addr) + socket.inet_aton(client_ip))
        self.sock.setblocking(blocking)

    def receive(self, size=65507,):
        try:
            pickled_data, addr = self.sock.recvfrom(size)
            data = pickle.loads(pickled_data)
            return (addr,data)
        except socket.error as e:
            return None

    def host_name(self):
        return socket.gethostname()

class Broadcaster:
    def __init__(self, self_name, producer_q, consumer_q, sleep_time=SLEEP_TIME):
        """
        a simple async send/receive handler for udp multicasting
        uses the queues to communicate with the caller
        """

        # use this name to ignore self-broadcasting
        self.name = self_name

        self.Producer = Producer()
        self.Consumer = Consumer()
        self.producer_q = producer_q
        self.consumer_q = consumer_q

        # handle of the thread, to check for life
        self.run_thread = None

        # sleep this much between sends
        self.sleep_time = sleep_time

    def _update(self):
        # try and receive any packages
        incoming = '<empty>'
        while incoming is not None:
            incoming = self.Consumer.receive()
            if incoming is not None:
                addr, data = incoming
                if data.get('owner') != self.name:
                    # ignore self-broadcasts
                    self.consumer_q.put_nowait(data)

        # send out the queued packages
        try:
            while True:
                outgoing = self.producer_q.get_nowait()
                self.producer_q.task_done()
                self.Producer.send(outgoing)
        except queue.Empty as e:
            pass

    def _run(self):
        while True:
            self._update()
            time.sleep(self.sleep_time)

    def start(self):
        self.run_thread = threading.Thread(target=self._run)
        self.run_thread.setDaemon(True)
        self.run_thread.start()


def make_broadcast(bcast_type, owner, **kwargs):
    """
    returns a standard broadcast dictionary
    give the expected arguments in kwargs

    accepted types:
        measurement -> index, (x,y,d) -> to be given to measurement_store.add_other
        gps -> pos, speed, heading -> to be given to body.set_xxx(xxx)
        filling -> measurements
        fill_request -> indices

        indices is a list of integers
        measurements is a list of [(i,x,y,d), ...] where i is the index
    """

    payload = None
    # owner is used by all to determine who the data belongs to
    if bcast_type == 'measurement':
        m_point = kwargs.get('m_point')
        index = kwargs.get('index')
        payload = {'m_point':m_point, 'index':index}

    if bcast_type == 'gps':
        pos = kwargs.get('pos')
        speed = kwargs.get('speed')
        heading = kwargs.get('heading')
        payload = {'pos':pos, 'speed':speed, 'heading':heading}

    if bcast_type == 'filling':
        measurements = kwargs.get('measurements')
        fill_to = kwargs.get('fill_to')
        payload = {'fill_to':fill_to, 'measurements':measurements}

    if bcast_type == 'fill_request':
        indices = kwargs.get('indices')
        fill_from = kwargs.get('fill_from')
        payload = {'fill_from':fill_from, 'indices':indices}

    return {'owner':owner, 'type':bcast_type, 'payload':payload}







"""
import udp
>>> # ok we create a producer using the default keyword arguments
>>> producer = udp.producer()
>>> # now lets make a client , using the default keyword arguments, so we can get the data
>>> consumer = udp.consumer()

>>> producer.send("Hello World!")
>>> # message sent to all clients
>>> # now when we want to get the message
>>> consumer.receive()
(('10.20.40.126', 1501), 'Hello World!')
>>> # what happens if we call again
>>> consumer.receive()
>>>
>>> # we get a None
>>> # You can put it in a loop and just call the receive until you get some action.
>>> # You can have multiple clients listening for the producer and each will be independant
>>> consumer2 = udp.consumer()
>>> producer.send([1,2,3,4,5])
>>> consumer.receive()
(('10.20.40.126', 1501), [1, 2, 3, 4, 5])
>>> consumer2 .receive()
(('10.20.40.126', 1501), [1, 2, 3, 4, 5])
>>> # you may notice the ip and port at the start of the message you can just ignore it for now
>>> msg = consumer2 .receive()
>>> if msg:
            ip_port, data = msg
"""
