import json

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from collections import defaultdict
from matplotlib.font_manager import FontProperties
from math import log


def memory_unit(memory_size):
    """
    This function translates the size from bytes to the most appropriate representation.

    Args:
        memory_size (int): The input memory size in bytes.

    Returns:
        int, string: new size, unit name.

    """
    if memory_size // (1024**3) != 0:
        memory_size /= 1024**3
        return memory_size, 'GiB'
    if memory_size // (1024**2) != 0:
        memory_size /= 1024**2
        return memory_size, 'MiB'
    if memory_size // 1024 != 0:
        memory_size /= 1024
        return memory_size, 'KiB'
    return memory_size, 'B'


def visualize(json_file):
    """
    This script allows you to see how the allocated memory is being reused and
    for what types of tensors (input, output, intermediate calculations).

    To use it, you need to run cli_applier with the "--memory-stats" parameter and
    paste the path to the json file into the "path_to_json_file" variable.

    The script will output an image on which the y-axis is responsible for the size of the blocks,
    relative to the beginning of each block, and the x-axis for the time.

    Args:
        json_file (str): The path to json file.

    Returns:
        figure, axes.

    """
    fig, ax = plt.subplots(figsize=(18, 10))

    ax.set_xlabel('Time')
    ax.set_ylabel('Blocks size')
    ax.set_title(json_file)

    with open(json_file) as json_file:
        json_data = json.load(json_file)

    results = defaultdict(list)

    sum_size = 0
    max_time = 0
    for i in range(len(json_data)):
        max_time = max(max_time, json_data[i]['LifeTimeEnd'])

        if json_data[i]['StorageBlockIdx'] not in results:
            sum_size += json_data[i]['StorageBlockUsedBytes']

        results[json_data[i]['StorageBlockIdx']].append(
            [
                json_data[i]['TensorType'],
                json_data[i]['LifeTimeBegin'],
                json_data[i]['LifeTimeEnd'],
                json_data[i]['StorageBlockUsedBytes'],
                json_data[i]['StorageBlockMemoryOffsetBytes'],
            ]
        )

    print(f'Blocks count: {len(results)}')
    normal_size, size_unit = memory_unit(sum_size)
    print(f'Memory size: {round(normal_size)}{size_unit}')

    white_space = 0
    for block_index, params in results.items():
        k = 0
        prev_size = 0
        for tensor_type, life_time_begin, life_time_end, tensor_size, memory_offset in params:
            if tensor_size:
                color = 'orange' if memory_offset == 0 else "chocolate"  # Paint virtual blocks darker
                if tensor_type == 'input':
                    color = 'green'
                elif tensor_type == 'output':
                    color = 'red'

                if k == 0:
                    prev_size = max(log(tensor_size), 10)
                    ax.broken_barh([(0, max_time)], (white_space, log(tensor_size)), facecolors='purple')

                log_memory_offset = log(memory_offset) if memory_offset > 0 else 0
                ax.broken_barh(
                    [(life_time_begin, life_time_end - life_time_begin)],
                    (white_space + log_memory_offset, log(tensor_size + memory_offset) - log_memory_offset),
                    facecolors=color,
                )
            k += 1

        white_space += prev_size + 15

    orange_patch = mpatches.Patch(color='orange', label='intermediate')
    chocolate_patch = mpatches.Patch(color='chocolate', label='intermediate (virtual block)')
    green_patch = mpatches.Patch(color='green', label='input')
    red_patch = mpatches.Patch(color='red', label='output')
    purple_patch = mpatches.Patch(color='purple', label='free memory')

    font_p = FontProperties()
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        prop=font_p,
        handles=[orange_patch, chocolate_patch, green_patch, red_patch, purple_patch],
    )
    fig.tight_layout()

    plt.show()

    return fig, ax
