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


def estimate_mutable_tensors_model(
    model: onnx.ModelProto,
    max_symbolic_var_params: dict[str, int],
    memory: DeviceMemory,
    logs_enabled: bool = False
) -> dict[str, TensorInfo]:
    graph = Graph(model, memory, set(max_symbolic_var_params.keys()))

    total_memory_estimated = 0

    total_nodes = len(graph.storage.nodes)
    pool = Pool(total_nodes, graph.model_memory)

    tensors_info: List[MutableTensorInfo] = graph.find_optimal_strategy(max_symbolic_var_params, logs_enabled)
    # tensors_info: List[MutableTensorInfo] = graph.straight_find_optional_strategy(max_symbolic_var_params, logs_enabled)

    tensors_info.sort()

    for info in tensors_info:
        total_memory_estimated += search_in_pool(info, pool, total_memory_estimated)
    
    return total_memory_estimated, tensors_info
