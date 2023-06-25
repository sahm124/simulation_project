from dataclasses import dataclass


def host1(x, simulation_time) -> [Event]:
    pass

@dataclass
class Event:
    created_at: float
    priority: int
    is_drop: bool = None
    queue_time: float = None
    service_time: float = None


@dataclass
class Processor:
    free_time: float = 0
    is_busy: bool = False
    to_busy: bool = 0


class Router:

    def __init__(self, processors_num, service_policy, y, simulation_time, event_table: [Event]):
        self.processors_num = processors_num
        self.queue = Fifo() if service_policy == "FIFO" else Wrr() if service_policy == "WRR" else Npps()
        self.event_table = event_table
        self.y = y
        self.simulation_time = simulation_time

    def arrive_packet(self, packet: tuple):
        self.queue.add(packet)

    def service_packet(self, packet):
        pass

    def generate_service_time(self):
        pass

    def run(self):
        time=0
        processors = [Processor in range(self.processors_num)]
        while(time<=self.simulation_time):
            pass

class Queue:

    def __init__(self, limit):
        self.limit = limit

    def add(self, packet: tuple):
        pass

    def call(self):
        pass


class Fifo(Queue):

    def call(self):
        pass

    def add(self):
        pass


class Wrr(Queue):

    def call(self):
        pass

    def add(self):
        pass


class Npps(Queue):
    def call(self):
        pass

    def add(self):
        pass


def simulation(PROCESSORS_NUM, SERVICE_POLICY, X, Y, T):
    event_table = [Event]

    event_table = host1(x=X, simulation_time=T)

    router = Router(event_table=event_table, processors_num=PROCESSORS_NUM, service_policy=SERVICE_POLICY, y=Y, simulation_time=T)
    router.run()
    print(event_table)


import math


class CLCG:
    def __init__(self, M, a, c, x0):
        self.M = M
        self.a = a
        self.c = c
        self.x = x0

    def rand(self):
        self.x = (self.a * self.x + self.c) % self.M
        return self.x / self.M


def exponential_generator(y, M, a, c, x0):
    clcg = CLCG(M, a, c, x0)
    while True:
        u = clcg.rand()  # generate a random number between 0 and 1 using CLCG
        x = -math.log(1 - u) / y  # apply the Inverse-Transform Technique
        yield x

gen = exponential_generator(2, 2**31 - 1, 1103515245, 12345, 123456789)
for i in range(10):
    x = next(gen)
    print(x)