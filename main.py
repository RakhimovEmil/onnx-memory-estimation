import argparse
from pathlib import Path

import onnx

from memory import DeviceMemory
from strategies import (
    estimate_mutable_tensors_naive, 
    estimate_mutable_tensors_greedy,
    estimate_mutable_tensors_model
)

def get_strategy(strategy_name: str):
    strategies = {
        "naive": estimate_mutable_tensors_naive,
        "greedy": estimate_mutable_tensors_greedy,
        "MODeL": estimate_mutable_tensors_model
    }
    if strategy_name not in strategies:
        raise NotImplementedError
    return strategies[strategy_name]

def main(args: argparse.Namespace):
    model = onnx.load_model(args.model_path)

    max_symbolic_param_values = {
        'bs': 8,
        'sq': 64,
        'B': 16,
        'T': 128,
        'batch_size': 8,
        'sequence_length': 64,
        'width': 224,
        'height': 224,
        'num_channels': 3
    }

    memory = DeviceMemory(alignment=args.alignment)

    strategy = get_strategy(args.strategy)
    mutable_memory_size, mutable_tensors_info = strategy(
        model, max_symbolic_param_values, memory
    )

    print("Mutable tensor estimation result:")
    # for tensor_name, tensor_info in mutable_tensors_info.items():
    #     print(f"{tensor_name}: {tensor_info}")
    for tensor_info in mutable_tensors_info:
        print(tensor_info.tensor_name, tensor_info.lifetime_begin, tensor_info.lifetime_end, tensor_info.estimated_size)
    print(f"Total memory: {mutable_memory_size} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to model with symbolic shapes",
    )
    parser.add_argument(
        "--alignment",
        type=int,
        default=256,  # Default GPU alignment
        help="Memory alignment of tensors in bytes",
    )
    parser.add_argument("--strategy", choices=["naive", "greedy", "MODeL"], default="naive")
    main(parser.parse_args())
