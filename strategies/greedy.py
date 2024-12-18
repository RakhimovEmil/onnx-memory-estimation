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
    max_symbolic_var_params: dict[str, int],
    memory: DeviceMemory,
    logs_enabled: False
) -> dict[str, TensorInfo]:
    graph = Graph(model, memory, set(max_symbolic_var_params.keys()))
    tensor_to_lifetime_end = graph.get_tensor_to_lifetime_end()

    tensors_info: List[MutableTensorInfo] = list()
    total_memory_estimated = 0

    total_nodes = len(graph.storage.nodes)
    pool = Pool(total_nodes, graph.model_memory)

    for idx, node in enumerate(graph.storage.nodes):
        for out_name in node.output_names:
            lifetime_end = tensor_to_lifetime_end[out_name]
            tensors_info.append(MutableTensorInfo(out_name, idx, lifetime_end))
            graph.estimate_single_tensor(tensors_info[-1], max_symbolic_var_params)

    #! раскомментировать если нужно добавить input
    # for input_name in graph.storage.input_names:
    #     tensor_value = graph.storage.values.get(input_name, None)

    #     if isinstance(tensor_value, MutableTensor):
    #         tensors_info.append(MutableTensorInfo(input_name, 0, total_nodes))
    #         graph.estimate_single_tensor(tensors_info[-1], max_symbolic_var_params)

    tensors_info.sort()

    for info in tensors_info:
        total_memory_estimated += search_in_pool(info, pool, total_memory_estimated)
    
    return total_memory_estimated, tensors_info
