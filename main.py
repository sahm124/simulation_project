

def host1(x, simulation_time) -> [Event]:
    pass

@dateclass
class Event:
    created_at: float
    priority: int
    is_drop: bool = None
    queue_time: float = None
    service_time: float = None


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