from memory import DeviceMemory
from strategies import (
    estimate_mutable_tensors_naive, 
    estimate_mutable_tensors_greedy,
    estimate_mutable_tensors_model
)
from visualization import visualize

import argparse
import onnx

from pathlib import Path

def get_strategy(strategy_name: str):
    strategies = {
        "naive": estimate_mutable_tensors_naive,
        "greedy": estimate_mutable_tensors_greedy,
        "MODeL": estimate_mutable_tensors_model
    }
    if strategy_name not in strategies:
        raise NotImplementedError
    return strategies[strategy_name]

max_symbolic_var_params = {
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


def get_strategy(strategy_name: str):
    strategies = {
        'naive': estimate_mutable_tensors_naive,
        'greedy': estimate_mutable_tensors_greedy,
        'MODeL': estimate_mutable_tensors_model
    }
    if strategy_name not in strategies:
        raise NotImplementedError
    return strategies[strategy_name]


def main(args: argparse.Namespace):
    model = onnx.load_model(args.model_path)

    memory = DeviceMemory(alignment=args.alignment)

    strategy = get_strategy(args.strategy)
    mutable_memory_size, mutable_tensors_info = strategy(
        model, max_symbolic_var_params, memory, args.verbose
    )

    if args.verbose:
        print('Mutable tensor estimation result:')
        for tensor_info in mutable_tensors_info:
            print(tensor_info.tensor_name, tensor_info.lifetime_begin, tensor_info.lifetime_end, tensor_info.estimated_size) # добавить адрес 
    
    if args.visualize:
        visualize(args.model_path, args.strategy, mutable_tensors_info)
        
    print(f'\nTotal memory: {mutable_memory_size} bytes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-path',
        type=Path,
        required=True,
        help='Path to model with symbolic shapes',
    )
    parser.add_argument(
        '--alignment',
        type=int,
        default=256,  # Default GPU alignment
        help='Memory alignment of tensors in bytes',
    )
    parser.add_argument('--strategy', choices=['naive', 'greedy', 'MODeL'], default='naive')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    main(parser.parse_args())
