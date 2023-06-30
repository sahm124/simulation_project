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

    def __init__(self, processors_num, service_policy, y, simulation_time, event_table: [Event]):
        self.processors_num = processors_num
        self.queue = Fifo() if service_policy == "FIFO" else Wrr() if service_policy == "WRR" else NPPS()
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
        min_time_of_processors = 0
        processor_availability = True
        process_availability = False
        min_arrive = self.event_table[0].created_at
        i = 0
        processors = [Processor in range(self.processors_num)]
        while (time <= self.simulation_time):
            if processor_availability and process_availability:
                for processor in processors:
                    if (not processor.is_busy) and (self.queue.not_empty) :
                        timer = self.queue.timer()
                        process_from_queue = self.queue.call()
                        processor.is_busy = True
                        processor.to_busy = time + timer
                        process_from_queue.remaining = process_from_queue.remaining - timer
                        if process_from_queue.remaining == 0:
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
                for processor in processors:
                    if not processor.is_busy:
                        processor.free_time = processor.free_time + (min_arrive - time)
                self.queue.update_mean_size(min_arrive)
                self.queue.add(self.event_table[i])
                time = min_arrive
                i = i+1
            else:
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
                        if processor.to_busy<miner:
                            miner = processor.to_busy
                self.queue.update_mean_size(time)



class Queue:
    def __init__(self, limit):
        self.limit = limit
        self.buffer = deque()

    def add(self, packet: Event):
        if len(self.buffer) < self.limit:
            self.buffer.append(packet)
        packet.is_drop = True

    def call(self):
        if not self.buffer:
            return None
        return self.buffer.popleft(), None


class Fifo(Queue):
    def __init__(self, limit):
        self.limit = limit
        self.buffer = deque()
        self.meaner = 0.0
        self.averege_count = 0.0
        self.q_time = 0.0

    def add(self, packet):
        super().add(packet)

    def call(self):
        return super().call()

    def update_mean_size(time):
        self.meaner = self.meaner + ((time - self.q_time)*len(self.buffer))
        self.q_time = time

    def calculate_averege(time):
        self.averege_count = self.meaner/time

    def timer(self):
        return self.buffer[0].remaining

    def is_empty(self):
        if len(self.buffer) != 0:
            return True
        return False



class NPPS(Queue):
    def __init__(self, limit):
        self.limit = limit
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
                self.q2.append()
            else:
                self.q3.append()

    def call(self):
        if len(self.q1)>0:
            return self.q1.popleft()
        elif len(self.q2)>0:
            return self.q2.popleft()
        else:
            return self.q3.popleft()

    def timer(self):
        if len(self.q1) > 0:
            return self.q1[0].remaining
        elif len(self.q2) > 0:
            return self.q2[0].remaining
        else:
            return self.q3[0].remaining

    def is_empty(self):
        if len(self.q1) > 0:
            return True
        elif len(self.q2) > 0:
            return True
        elif len(self.q3) > 0:
            return True
        return False

    def update_mean_size(time):
        self.meaner = self.meaner + ((time - self.q_time)*(len(self.q1)+len(self.q2)+len(self.q3)))
        self.q_time = time

    def calculate_averege(time):
        self.averege_count = self.meaner/time


class Wrr(Queue):
    def __init__(self, limit):
        self.limit = limit
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
        self.remaining_time = 3

    def add(self, packet):
        if packet.priority == 1:
            if len(self.q1) < self.limit:
                self.q1.append(packet)
        elif packet.priority == 2:
            if len(self.q2) < self.limit:
                self.q2.append()
        elif packet.priority == 3:
            if len(self.q3) < self.limit:
                self.q3.append()

    def call(self):
        if self.which_q == 1:
            if len(self.q1) > 0:
                if self.remaining_time >= self.q1[0].remaining:
                    self.remaining_time = self.remaining_time - self.q1[0].remaining
                    return self.q1.pop()
                else :
                    self.remaining_time = 2
                    self.which_q = 2
                    return self.q1[0]
            else:
                self.remaining_time = 2
                self.which_q = 2
                return self.call()

        elif self.which_q == 2:
            if len(self.q2) > 0:
                if self.remaining_time >= self.q2[0].remaining:
                    self.remaining_time = self.remaining_time - self.q2[0].remaining
                    return self.q2.pop()
                else:
                    self.remaining_time = 1
                    self.which_q = 3
                    return self.q2[0]
            else:
                self.remaining_time = 1
                self.which_q = 3
                return self.call()

        elif self.which_q == 3:
            if len(self.q3) > 0:
                if self.remaining_time >= self.q3[0].remaining:
                    self.remaining_time = self.remaining_time - self.q3[0].remaining
                    return self.q3.pop()
                else:
                    self.remaining_time = 3
                    self.which_q = 1
                    return self.q3[0]
            else:
                self.remaining_time = 3
                self.which_q = 1
                return self.call()


    def timer(self):
        if self.remaining_time == 0:
            if self.which_q ==1:
                self.which_q ==2
                self.remaining_time=2
            elif self.which_q ==2:
                self.which_q == 3
                self.remaining_time = 1
            else:
                self.which_q == 1
                self.remaining_time = 3

        if self.which_q == 1:
            if len(self.q1) > 0:
                if self.remaining_time >= self.q1[0].remaining:
                    return self.q1[0].remaining
                else:
                    return self.remaining_time
            elif len(self.q2) > 0:
                if 2 >= self.q2[0].remaining:
                    return self.q2[0].remaining
                else:
                    return 2
            elif len(self.q3) > 0:
                if 1 >= self.q2[0].remaining:
                    return self.q2[0].remaining
                else:
                    return 1

        elif self.which_q == 2:
            if len(self.q2) > 0:
                if self.remaining_time >= self.q2[0].remaining:
                    return self.q2[0].remaining
                else:
                    return self.remaining_time
            elif len(self.q3) > 0:
                if 1 >= self.q3[0].remaining:
                    return self.q3[0].remaining
                else:
                    return 1
            elif len(self.q1) > 0:
                if 3 >= self.q1[0].remaining:
                    return self.q1[0].remaining
                else:
                    return 3

        elif self.which_q == 3:
            if len(self.q3) > 0:
                if self.remaining_time >= self.q3[0].remaining:
                    return self.q3[0].remaining
                else:
                    return self.remaining_time
            elif len(self.q1) > 0:
                if 3 >= self.q1[0].remaining:
                    return self.q1[0].remaining
                else:
                    return 3
            elif len(self.q2) > 0:
                if 2 >= self.q2[0].remaining:
                    return self.q2[0].remaining
                else:
                    return 2

    def is_empty(self):
        if len(self.q1) > 0:
            return True
        elif len(self.q2) > 0:
            return True
        elif len(self.q3) > 0:
            return True
        return False

    def update_mean_size(time):
        self.meaner1 = self.meaner1 + ((time - self.q_time) * (len(self.q1)))
        self.meaner2 = self.meaner2 + ((time - self.q_time) * (len(self.q2)))
        self.meaner3 = self.meaner3 + ((time - self.q_time) * (len(self.q3)))
        self.q_time = time

    def calculate_averege(time):
        self.averege_count1 = self.meaner1 / time
        self.averege_count2 = self.meaner2 / time
        self.averege_count3 = self.meaner3 / time


def simulation(PROCESSORS_NUM, SERVICE_POLICY, X, Y, T):
    event_table = [Event]

    event_table = host1(x=X, simulation_time=T)

    router = Router(event_table=event_table, processors_num=PROCESSORS_NUM, service_policy=SERVICE_POLICY, y=Y,
                    simulation_time=T)
    router.run()
    print(event_table)


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
        event = Event(
            created_at=arrive,
            service_time=next(service_time_gen),
            priority=next(packet_gen),
        )
        events.append(event)
    return events
# gen = exponential_generator(2, 2**31 - 1, 2247445469, 12345, 123456789)
# for i in range(10):
#     x = next(gen)
#     print(x)
# z =[]
# gen = CLCG(M=2**31 - 1, a=22474454, c=123456, x0=123456789)
# print("uniform")
# for i in range(10):
#     x = gen.rand()
#     z.append(x)
#     print(x)
#
# import numpy as np
#
# # Generate some sample data
#
#
# # Calculate the variance of the data
# data_var = np.var(z)
#
# # Calculate the expected variance of a uniform distribution with the same range as the data
# uniform_var = (1/12) * (max(z) - min(z))**2
#
# # Check if the ratio of the actual variance to the expected variance is close to 1
# if abs(data_var / uniform_var - 1) < 0.1:
#     print("The data is likely from a uniform distribution.")
# else:
#     print("The data is not from a uniform distribution.")
#
# poisson_gen = poisson_generator(2.5, M=2**31-1, a=1103515245, c=12345, x0=1)
#
# for i in range(10):
#     x = next(poisson_gen)
#     print(x)
