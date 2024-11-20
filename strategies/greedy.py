import onnx

from typing import List

from memory import (
    DeviceMemory,
    TensorInfo, 
    Pool
)
from graph import Graph
from containers import (
    MutableTensor,
    MutableTensorInfo,
)

from memory import search_in_pool


def estimate_mutable_tensors_greedy(
    model: onnx.ModelProto,
    max_batch_params: dict[str, int],
    memory: DeviceMemory,
    logs_enabled: False
) -> dict[str, TensorInfo]:
    graph = Graph(model, memory, set(max_batch_params.keys()))
    last_input_user_indexes = graph.get_last_input_user_index()

    tensors_info: List[MutableTensorInfo] = list()
    total_memory_estimated = 0

    total_nodes = len(graph.storage.nodes)
    pool = Pool(total_nodes, graph.model_memory)

    for idx, node in enumerate(graph.storage.nodes):
        for out_name in node.output_names:
            death_time = last_input_user_indexes[out_name]
            tensors_info.append(MutableTensorInfo(out_name, idx, death_time))
            graph.estimate_single_tensor(tensors_info[-1], max_batch_params)

    # for input_name in graph.storage.input_names:
    #     tensor_value = graph.storage.values.get(input_name, None)

    #     if isinstance(tensor_value, MutableTensor):
    #         tensors_info.append(MutableTensorInfo(input_name, 0, total_nodes))
    #         graph.estimate_single_tensor(tensors_info[-1], max_batch_params)

    tensors_info.sort()

    for info in tensors_info:
        total_memory_estimated += search_in_pool(info, pool, total_memory_estimated)
    
    return total_memory_estimated, tensors_info
