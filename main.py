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


@dataclass
class Processor:
    free_time: float = 0
    is_busy: bool = False
    to_busy: bool = 0


def host1(x, simulation_time) -> [Event]:
    pass


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
        while(time<=self.simulation_time):
            if processor_availability and process_availability:
                for processor in processors:
                    if (not processor.is_busy) and (self.queue.not_empty) :
                        process_from_queue = self.queue.call()
                        processor.is_busy = True
                        generated_service_time = self.generate_service_time()
                        processor.to_busy = time + generated_service_time
                        process_from_queue.service_time = generated_service_time
                        process_from_queue.queue_time = time - process_from_queue.created_at
                miner = 1000
                processor_availability = False
                for processor in processors:
                    if (processor.to_busy < miner) and processor.is_busy:
                        miner = processor.to_busy
                    if not processor.is_busy:
                        processor_availability = True
                min_time_of_processors = miner
            elif (min_arrive < min_time_of_processors) and (i != -1):
                for processor in processors:
                    if not processor.is_busy:
                        processor.free_time = processor.free_time + (min_arrive - time)
                self.queue.add(self.event_table[i])
                time = min_arrive
                i = i+1
                if i < self.event_table.lenth()-1:
                    min_arrive = self.event_table[i].created_at
                else:
                    i = -1
            elif min_time_of_processors <= min_arrive:
                for processor in processors:
                    if not processor.is_busy:
                        processor.free_time = processor.free_time + (min_time_of_processors - time)
                for processor in processors:
                    if processor.to_busy == min_time_of_processors:
                        processor.is_busy = False
                time = min_time_of_processors




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
        return self.buffer.popleft()


class Fifo(Queue):
    def add(self, packet):
        super().add(packet)

    def call(self):
        return super().call()


class Wrr(Queue):
    def __init__(self, limit, weights):
        super().__init__(limit)
        self.weights = deque(weights)

    def add(self, packet):
        if len(self.buffer) < self.limit:
            self.buffer.append((packet, self.weights[0]))
            self.weights.rotate(-1)

    def call(self):
        if not self.buffer:
            return None
        packet, weight = self.buffer.popleft()
        self.weights.append(weight)
        return packet


class Npps(Queue):
    def __init__(self, limit):
        super().__init__(limit)
        self.priorities = {1: deque(), 2: deque(), 3: deque()}

    def add(self, packet, priority):
        if len(self.buffer) < self.limit:
            self.priorities[priority].append(packet)

    def call(self):
        for priority in [1, 2, 3]:
            if self.priorities[priority]:
                packet = self.priorities[priority].popleft()
                return packet
        return None


def simulation(PROCESSORS_NUM, SERVICE_POLICY, X, Y, T):
    event_table = [Event]

    event_table = host1(x=X, simulation_time=T)

    router = Router(event_table=event_table, processors_num=PROCESSORS_NUM, service_policy=SERVICE_POLICY, y=Y, simulation_time=T)
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
        u = clcg.rand()  # generate a random number between 0 and 1 using CLCG
        x = -math.log(1 - u) / y  # apply the Inverse-Transform Technique
        yield x

def poisson_generator(lam, M, a, c, x0):
    clcg = CLCG(M, a, c, x0)
    n = 1000  # set the number of trials for the binomial distribution
    p = lam / n  # set the probability of success for the binomial distribution
    while True:
        u = clcg.rand()  # generate a random number between 0 and 1 using CLCG
        x = -math.log(1 - u) / lam  # generate an exponential random variable
        y = 0
        for i in range(n):
            if clcg.rand() < p:
                y += 1
        if x <= y / p:  # check if x is less than or equal to y/p
            yield y
        else:
            continue

def arrival_times(lambda_val, num_events, M=2**32, a=1103515245, c=12345, x0=1):
    # Generate inter-arrival times using the poisson_generator function
    inter_arrival_gen = poisson_generator(lambda_val, M, a, c, x0)
    inter_arrival_times = [next(inter_arrival_gen) for i in range(num_events)]

    # Accumulate the inter-arrival times to get the arrival times
    arrival_times = [sum(inter_arrival_times[:i+1]) for i in range(num_events)]

    return arrival_times

gen = exponential_generator(2, 2**31 - 1, 2247445469, 12345, 123456789)
for i in range(10):
    x = next(gen)
    print(x)
z =[]
gen = CLCG(M=2**31 - 1, a=22474454, c=123456, x0=123456789)
print("uniform")
for i in range(10):
    x = gen.rand()
    z.append(x)
    print(x)

import numpy as np

# Generate some sample data


# Calculate the variance of the data
data_var = np.var(z)

# Calculate the expected variance of a uniform distribution with the same range as the data
uniform_var = (1/12) * (max(z) - min(z))**2

# Check if the ratio of the actual variance to the expected variance is close to 1
if abs(data_var / uniform_var - 1) < 0.1:
    print("The data is likely from a uniform distribution.")
else:
    print("The data is not from a uniform distribution.")

poisson_gen = poisson_generator(2.5, M=2**31-1, a=1103515245, c=12345, x0=1)

for i in range(10):
    x = next(poisson_gen)
    print(x)
