

def host():
    pass

@dateclass
class Event:
    priority: int
    is_drop: bool
    created_at: float
    queue_time: float
    service_time: float

event_table = [Event]
#[priority, drop=0/1, created_at, queue_time, service_time]


class Router:

    def __init__(self, processors_num, service_policy):
        self.processors_num = processors_num
        self.queue = Fifo() if service_policy == "FIFO" else Wrr() if service_policy == "WRR" else Npps()

    def arrive_packet(self, packet: tuple):
        self.queue.add(packet)

    def service_packet(self, packet):
        pass

    def generate_service_time(self):
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
