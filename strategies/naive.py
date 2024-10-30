import onnx

from google.protobuf.json_format import MessageToDict
from typing import List

from containers import MutableTensorInfo
from expression import Expression
from memory import (
    TensorInfo, 
    DeviceMemory
)


class NaiveTensorMemoryEstimator:
    def __init__(
        self,
        model: onnx.ModelProto,
        max_symbolic_param_values: dict[str, int],
        device_memory: DeviceMemory,
    ) -> None:
        self.memory = device_memory
        self.num_nodes = len(model.graph.node)
        self.total_allocated_bytes = 0
        self.max_symbolic_param_values = max_symbolic_param_values

    def _eval_dim(self, dim: onnx.TensorShapeProto.Dimension) -> int:
        dim_dict = MessageToDict(dim)
        if "dimValue" in dim_dict:
            value = Expression(
                dim_dict["dimValue"], self.max_symbolic_param_values.keys()
            ).evaluate(self.max_symbolic_param_values)
        elif "dimParam" in dim_dict:
            value = Expression(
                dim_dict["dimParam"], self.max_symbolic_param_values.keys()
            ).evaluate(self.max_symbolic_param_values)
        else:
            raise ValueError(f"Dimension is neither a constant nor expression")

        truncated = int(value)
        if truncated != value:
            raise ValueError(f"Expression returned non-integer result {value}")

        return truncated

    def estimate_single_tensor(self, tensor: onnx.TensorProto) -> TensorInfo:
        dims = [self._eval_dim(dim) for dim in tensor.shape.dim]

        result = TensorInfo(
            dims=dims,
            dtype=onnx.TensorProto.DataType.Name(tensor.elem_type),
            lifetime_begin=0,
            lifetime_end=self.num_nodes,  # Every tensor is alive from beginning to end
            memory_offset=self.total_allocated_bytes,
        )
        self.total_allocated_bytes += self.memory.get_tensor_size(result)
        return result


def estimate_mutable_tensors_naive(
    model: onnx.ModelProto,
    max_symbolic_param_values: dict[str, int],
    memory: DeviceMemory,
) -> dict[str, TensorInfo]:
    result = {}

    estimator = NaiveTensorMemoryEstimator(model, max_symbolic_param_values, memory)

    for input_ in model.graph.input:
        result[input_.name] = estimator.estimate_single_tensor(input_.type.tensor_type)

    for value in model.graph.value_info:
        result[value.name] = estimator.estimate_single_tensor(value.type.tensor_type)

    return estimator.total_allocated_bytes, result
