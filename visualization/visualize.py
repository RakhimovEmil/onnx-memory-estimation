from containers import MutableTensorInfo

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from typing import List

matplotlib.use('Agg') 

def visualize(strategy: str, tensor_infos: List[MutableTensorInfo]):
    events = []
    for tensor in tensor_infos:
        events.append((tensor.lifetime_begin, tensor.estimated_size / 1024**2))
        events.append((tensor.lifetime_end, -tensor.estimated_size / 1024**2))

    events.sort()

    times = []
    memory_usage = []
    current_memory = 0

    for time, size_change in events:
        times.append(time)
        memory_usage.append(current_memory)

        current_memory += size_change
        times.append(time)
        memory_usage.append(current_memory)

    if times[-1] != 0:
        times.append(times[-1] + 1)
        memory_usage.append(0)

    plt.figure(figsize=(10, 6))
    plt.title(f'tensor memory usage over graph steps | {strategy}')
    
    step = (max(times) >= 35) + (max(times) >= 70) + 1
    plt.xticks(np.arange(min(times), max(times) + 1, step))
    plt.yticks(np.linspace(min(memory_usage), max(memory_usage), 10))

    plt.xlabel('step')
    plt.ylabel('memory usage (MB)')
    plt.step(times, memory_usage, where='post', label='Memory Usage')
    
    plt.legend()
    plt.grid(True)
    plt.savefig('memory_usage.png')

    print('график сохранен в memory_usage.png')
