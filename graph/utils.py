import onnx

from google.protobuf.json_format import MessageToDict
from typing import Dict, List, Set, Union, Tuple

from containers import (
    ConstTensor,
    ConstTensorInfo,
    MutableTensor,
    NodeWrapper,
)

from expression import Expression


def evaluate_integer(expr: Expression, placeholders: Dict[str, int]) -> int:
    raw_result = expr.evaluate(placeholders)
    trunc_result = int(raw_result)

    if trunc_result != raw_result:
        raise ValueError(f'Expression {expr} returned non-integer result {raw_result}')
    return trunc_result

def evaluate_tensor_dimensions(dim_exprs: List[Expression], placeholders: Dict[str, int]) -> List[int]:
    return [evaluate_integer(dim_expr, placeholders) for dim_expr in dim_exprs]


ValueType = Union[Expression, List[Expression], ConstTensor, MutableTensor]

def _load_symbolic_node(node: onnx.NodeProto, placeholder_names: Set[str]) -> Tuple[str, ValueType]:
    if len(node.input) != 0 or len(node.output) != 1:
        raise ValueError(f'A symbolic node is expected to have no inputs and a single output, not {len(node.input)} and {len(node.output)}')

    symb_expr = next((a for a in node.attribute if a.name == 'symbolic_expr'), None)

    if symb_expr is None:
        raise AttributeError(f'Failed to find a valid symbolic_expr attribute for a symbolic node')
    
    output_name = node.output[0]
    value: ValueType = None

    if hasattr(symb_expr, 's'):
        value = Expression(symb_expr.s.decode('utf-8'), placeholder_names)
    elif len(symb_expr.strings) > 0:
        dims: List[Expression] = list()
        for dim_string in symb_expr.strings:
            dims.append(Expression(dim_string.decode('utf-8'), placeholder_names))
        value = dims
    elif symb_expr.type != onnx.AttributeProto.AttributeType.Value('STRING'):
        raise ValueError(f'Expected a string or a sequence of strings in symbolic_expr attribute, got {onnx.TensorProto.DataType.Name(symb_expr.type)}')

    return output_name, value if value is not None else list()

def _load_regular_node(node: onnx.NodeProto):
    # print(node.output)
    return NodeWrapper(
        node.name,
        node.op_type,
        list(filter(str, reversed(node.input))),
        list(node.output),
    )

def load_nodes(model: onnx.ModelProto, placeholder_names: Set[str]) -> Tuple[Dict[str, ValueType], List[NodeWrapper]]:
    values: Dict[str, ValueType] = dict()
    nodes: List[NodeWrapper] = list()

    for node in model.graph.node:
        # print(f'node: {node.name}')
        if node.domain == 'ru.yandex.ynmt' and node.op_type == 'EvaluateSymbolicExpression': # ? ни разу не встречалось, видимо артефакт
            # print(1)
            name, value = _load_symbolic_node(node, placeholder_names)
            values[name] = value
        elif hasattr(node, 'domain') or node.op_type != 'Constant':
            # print(2)
            nodes.append(_load_regular_node(node))
    
    return values, nodes


def _load_const_tensor(tensor: ConstTensorInfo):
    return ConstTensor(tensor.dims)


def load_const_tensors(model: onnx.ModelProto) -> Dict[str, ValueType]:
    infos: Dict[str, ConstTensorInfo] = dict()

    for init in model.graph.initializer:
        if init.name in infos:
            raise ValueError(f'Initializer {init.name} does not have a unique name')
        infos[init.name] = ConstTensorInfo(init, False)

    for node in model.graph.node:
        if node.op_type != 'Constant' or hasattr(node, 'domain'):
            continue

        tensor_found = False

        for attr in node.attribute:
            if attr.name != 'value' or not hasattr(attr, 't'):
                continue
            
            if len(node.output) != 1:
                raise ValueError(f'Constant node {node.name} has outputs instead of a single one')

            output_name = node.output[0]

            if output_name in infos:
                raise ValueError(f'Output {output_name} does not have an unique name')

            infos[output_name] = ConstTensorInfo(attr.t, False)
            tensor_found = True

        if not tensor_found:
            raise AttributeError(f'Constant node {node.name} does not have a tensor value')

    values: Dict[str, ValueType] = dict()

    for name, info in infos.items():
        values[name] = _load_const_tensor(info)

    return values


def _load_values_from(values, placeholder_names) -> Dict[str, ValueType]:
    result_values: Dict[str, ValueType] = dict()

    for value in values:
        # print(f'name: {value}')
        if not hasattr(value.type, 'tensor_type'):
            raise ValueError('Non tensor-type values are not supported')

        tensor = value.type.tensor_type

        expressions: List[Expression] = list()
        for idx, dim in enumerate(tensor.shape.dim):
            dim_dict = MessageToDict(dim)
            if 'dimValue' in dim_dict:
                expressions.append(Expression(dim_dict['dimValue'], placeholder_names))
            elif 'dimParam' in dim_dict:
                expressions.append(Expression(dim_dict['dimParam'], placeholder_names))
            else:
                raise ValueError(f'Dimension #{idx} of value {value.name} is neither a constant nor expression')
        
        result_values[value.name] = MutableTensor(expressions, onnx.TensorProto.DataType.Name(tensor.elem_type))
    
    return result_values

def load_values(model: onnx.ModelProto, placeholder_names: Set[str]) -> Dict[str, ValueType]:
    values: Dict[str, ValueType] = dict()

    values.update(_load_values_from(model.graph.input, placeholder_names))
    values.update(_load_values_from(model.graph.output, placeholder_names))
    values.update(_load_values_from(model.graph.value_info, placeholder_names))

    return values
