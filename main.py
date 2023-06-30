from dataclasses import dataclass
import math
from collections import deque


@dataclass
class Event:
    created_at: float
    priority: int
    is_drop: bool = None
    queue_time: float = None
    service_time: float = None
    is_done = False
    remaining: float = None


@dataclass
class Processor:
    free_time: float = 0
    is_busy: bool = False
    to_busy: bool = 0


def host1(x, simulation_time) -> [Event]:
    raise NotImplementedError


class Router:

    def __init__(self, processors_num, service_policy, y, simulation_time, event_table: [Event], limit1, limit2, limit3):
        self.processors_num = processors_num
        self.queue = Fifo(limit1,limit2,limit3) if service_policy == "FIFO" else Wrr(limit1,limit2,limit3) if service_policy == "WRR" else NPPS(limit1,limit2,limit3)
        self.service_p = service_policy
        self.event_table = event_table
        self.y = y
        self.simulation_time = simulation_time

    def arrive_packet(self, packet: tuple):
        self.queue.add(packet)

    def service_packet(self, packet):
        pass

    def generate_service_time(self) -> float:
        pass

    def run(self):
        time = 0
        min_time_of_processors = 1000
        processor_availability = True
        process_availability = False
        min_arrive = self.event_table[0].created_at
        i = 0
        processors = [Processor() for i in range(self.processors_num)]
        miner = 1000

        while (time <= self.simulation_time):
            print("min_time_of_processors", min_time_of_processors)
            print("min_arrive", min_arrive)
            print("queue", self.queue.buffer)
            if processor_availability and self.queue.not_empty():
                print('processor_availability and process_availability')
                print(time)
                for processor in processors:
                    if self.queue.not_empty():
                        if not processor.is_busy:
                            process_from_queue = self.queue.call()
                            processor.is_busy = True
                            processor.to_busy = time + process_from_queue.service_time
                            process_from_queue.is_done = True
                            process_from_queue.queue_time = time - process_from_queue.created_at
                miner = 1000
                processor_availability = False
                for processor in processors:
                    if (processor.to_busy < miner) and processor.is_busy:
                        miner = processor.to_busy
                    if not processor.is_busy:
                        processor_availability = True
                min_time_of_processors = miner
            elif min_arrive < min_time_of_processors:
                print('min_arrive < min_time_of_processors')
                print(time)
                for processor in processors:
                    if not processor.is_busy:
                        processor.free_time = processor.free_time + (min_arrive - time)
                self.queue.update_mean_size(min_arrive)
                self.queue.add(self.event_table[i])
                time = min_arrive
                min_arrive = self.event_table[i+1].created_at
                i = i+1
                miner= 1000
                for processor in processors:
                    if (processor.to_busy < miner) and processor.is_busy:
                        miner = processor.to_busy
                    if not processor.is_busy:
                        processor_availability = True
                min_time_of_processors = miner
            else:
                print('else')
                print(time)
                for processor in processors:
                    if not processor.is_busy:
                        processor.free_time = processor.free_time + (min_time_of_processors - time)
                for processor in processors:
                    if processor.to_busy == min_time_of_processors:
                        processor.is_busy = False
                time = min_time_of_processors
                miner = 1000
                for processor in processors:
                    if processor.is_busy:
                        if processor.to_busy < miner:
                            miner = processor.to_busy
                min_time_of_processors = miner
                self.queue.update_mean_size(time)

        self.queue.update_mean_size(self.simulation_time)
        self.queue.calculate_averege(self.simulation_time)

        if self.service_p == "FIFO":
            print("averge count of the queue is :")
            print(self.queue.averege_count)
        elif self.service_p == "WRR":
            print("averge count of the first queue is :")
            print(self.queue.averege_count1)
            print("averge count of the second queue is :")
            print(self.queue.averege_count2)
            print("averge count of the third queue is :")
            print(self.queue.averege_count3)
        else:
            print("averge count of the queue is :")
            print(self.queue.averege_count)





class Queue:
    def __init__(self, limit):
        self.limit = limit
        self.buffer = deque()

    def add(self, packet: Event):
        if len(self.buffer) < self.limit:
            self.buffer.append(packet)
        else:
            packet.is_drop = True

    def call(self):
        if not self.buffer:
            return None
        return self.buffer.popleft()


class Fifo(Queue):
    def __init__(self, limit1, limit2, limit3):
        self.limit = limit1
        self.buffer = deque()
        self.meaner = 0.0
        self.averege_count = 0.0
        self.q_time = 0.0

    def add(self, packet):
        if len(self.buffer) < self.limit:
            self.buffer.append(packet)
        else:
            packet.is_drop = True

    def call(self):
        if not self.buffer:
            return None
        return self.buffer.popleft()

    def update_mean_size(self, time):
        self.meaner = self.meaner + ((time - self.q_time)*len(self.buffer))
        self.q_time = time

    def calculate_averege(self, time):
        self.averege_count = self.meaner/time

    def not_empty(self):
        if len(self.buffer) != 0:
            return True
        return False



class NPPS(Queue):
    def __init__(self, limit1, limit2, limit3):
        self.limit = limit1
        self.buffer = deque()
        self.meaner = 0.0
        self.averege_count = 0.0
        self.q_time = 0.0
        self.q1 = deque()
        self.q2 = deque()
        self.q3 = deque()

    def add(self, packet):
        if (len(self.q1)+len(self.q2)+len(self.q3)) < self.limit:
            if packet.priority == 1:
                self.q1.append(packet)
            elif packet.priority == 2:
                self.q2.append(packet)
            else:
                self.q3.append(packet)
        else:
            packet.is_drop = True

    def call(self):
        if len(self.q1)>0:
            return self.q1.popleft()
        elif len(self.q2)>0:
            return self.q2.popleft()
        else:
            return self.q3.popleft()

    def not_empty(self):
        if len(self.q1) > 0:
            return True
        elif len(self.q2) > 0:
            return True
        elif len(self.q3) > 0:
            return True
        return False

    def update_mean_size(self, time):
        self.meaner = self.meaner + ((time - self.q_time)*(len(self.q1)+len(self.q2)+len(self.q3)))
        self.q_time = time

    def calculate_averege(self, time):
        self.averege_count = self.meaner/time


class Wrr(Queue):
    def __init__(self, limit1, limit2, limit3):
        self.limit3 = limit3
        self.limit2 = limit2
        self.limit1 = limit1
        self.buffer = deque()
        self.meaner1 = 0.0
        self.averege_count1 = 0.0
        self.meaner2 = 0.0
        self.averege_count2 = 0.0
        self.meaner3 = 0.0
        self.averege_count3 = 0.0
        self.q_time = 0.0
        self.q1 = deque()
        self.q2 = deque()
        self.q3 = deque()
        self.which_q = 1
        self.remaining = 3

    def add(self, packet):
        if packet.priority == 1:
            if len(self.q1) < self.limit1:
                self.q1.append(packet)
        elif packet.priority == 2:
            if len(self.q2) < self.limit2:
                self.q2.append(packet)
        elif packet.priority == 3:
            if len(self.q3) < self.limit3:
                self.q3.append(packet)
        else:
            packet.is_drop=True

    def call(self):
        if self.remaining == 0 :
          if self.which_q ==1:
            self.which_q =2
            self.remaining = 2
          elif self.which_q ==2:
            self.which_q =3
            self.remaining = 1
          elif self.which_q ==3:
            self.which_q =1
            self.remaining = 3

        if self.which_q == 1:
            if len(self.q1) > 0:
              self.remaining = self.remaining -1
              return self.q1.popleft()
            else:
                self.which_q = 2
                self.remaining = 2
                return self.call

        elif self.which_q == 2:
            if len(self.q2) > 0:
              self.remaining = self.remaining -1
              return self.q2.popleft()
            else:
                self.which_q = 3
                self.remaining = 1
                return self.call

        elif self.which_q == 3:
            if len(self.q3) > 0:
              self.remaining = self.remaining -1
              return self.q3.popleft()
            else:
                self.which_q = 1
                self.remaining = 3
                return self.call


    def not_empty(self):
        if len(self.q1) > 0:
            return True
        elif len(self.q2) > 0:
            return True
        elif len(self.q3) > 0:
            return True
        return False

    def update_mean_size(self, time):
        self.meaner1 = self.meaner1 + ((time - self.q_time) * (len(self.q1)))
        self.meaner2 = self.meaner2 + ((time - self.q_time) * (len(self.q2)))
        self.meaner3 = self.meaner3 + ((time - self.q_time) * (len(self.q3)))
        self.q_time = time

    def calculate_averege(self, time):
        self.averege_count1 = self.meaner1 / time
        self.averege_count2 = self.meaner2 / time
        self.averege_count3 = self.meaner3 / time


def simulation(PROCESSORS_NUM, SERVICE_POLICY, X, Y, T, limit1,limit2,limit3):
    event_table = [Event]
    event_table = host(simulation_time=2*T, x=X, y=Y)

    router = Router(event_table=event_table, processors_num=PROCESSORS_NUM, service_policy=SERVICE_POLICY, y=Y,
                    simulation_time=T, limit1= limit1, limit2=limit2, limit3=limit3)
    router.run()

    return event_table


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
        u = clcg.rand()
        x = -math.log(1 - u) / y  # Inverse-Transform Technique
        yield x


def poisson_generator_by_binomial(lam, M, a, c, x0):
    clcg = CLCG(M, a, c, x0)
    n = 1000
    p = lam / n
    while True:
        u = clcg.rand()
        x = -math.log(1 - u) / lam
        y = 0
        for i in range(n):
            if clcg.rand() < p:
                y += 1
        if x <= y / p:
            yield y
        else:
            continue


def poisson_time_arrival_generator(lam, M, a, c, x0, total_time):
    exp_gen = exponential_generator(1 / lam, M, a, c, x0)
    arrival_times = []
    t = 0
    while t < total_time:
        interrval_time = next(exp_gen)
        t += interrval_time
        arrival_times.append(t)
    return arrival_times


def priority_packet(M, a, c, x0):
    clcg = CLCG(M, a, c, x0)
    while True:
        u = clcg.rand()
        if (u <= 0.2) and (u >= 0):
            yield 1
        elif u <= 0.5:
            yield 2
        elif u <= 1:
            yield 3
        else:
            raise Exception("Not uniform generated")

def host(simulation_time, x, y):
    M = 2 ** 31 - 1
    a = 1103515245
    c = 12345
    x0 = 1
    arrival_time = poisson_time_arrival_generator(lam=x, M=M, a=a, c=c, x0=x0, total_time=simulation_time)
    packet_gen = priority_packet(M=M, a=a, c=c, x0=x0)
    service_time_gen = exponential_generator(y=y, M=M, a=a, c=c, x0=x0)
    events = []
    for arrive in arrival_time:
        service_time = next(service_time_gen)
        event = Event(
            created_at=arrive,
            service_time=service_time,
            priority=next(packet_gen),
            remaining=service_time,
        )
        events.append(event)
    return events
PROCESSORS_NUM = 1
SERVICE_POLICY = 'NPPS'
X = 1
Y = 0.1
T = 10
limit1 = 0
limit2 = 1
limit3 = 1
final_event_table = simulation(PROCESSORS_NUM=PROCESSORS_NUM, SERVICE_POLICY=SERVICE_POLICY, X=X, Y=Y, T=T, limit1=limit1,limit2=limit2,limit3=limit3)
for i in final_event_table:
    print(i)