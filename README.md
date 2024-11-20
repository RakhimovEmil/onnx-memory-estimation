# ONNX Memory Estimation

A testbed for estimating the performance of tensor allocation algorithms for ONNX neural networks.
A naive strategy is implemented as a baseline, where each tensor resides in the memory for the whole graph inference time.
Currently, only mutable tensors (e.g. activations) are estimated.
Constant tensors, such as weights, are assumed to be constantly resident in the device memory.

Each tensor is described by its dimensions, data type, memory offset and lifetime.
An ONNX graph is inferred in some sequential node order, i.e. no two nodes execute concurrently.
Thus, the tensor's lifetime can be described by two integers: `lifetime_begin` and `lifetime_end`.
These numbers are the sequential numbers of the node that first uses the tensor and the node that last uses it, respectively.
In theory, we can store two tensors in the same memory region if their lifetimes do not overlap.
However, the naive strategy does not support this approach.

## Usage

1. Prepare the environment:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install -r requirments.txt
    ```

1. Download a test ONNX AlexNet model and run `symbolic_shape_infer.py`:

    ```bash
    python3 onnx/download_models.py
    python3 tools/symbolic_shape_infer.py --input onnx/alexnet.onnx --output onnx/alexnet_symbolic_shapes.onnx
    ```

    Symbolic shape inference is required to estimate the variable dimensions for each tensor as symbolic expressions (e.g. `[batch_size*2, 1, 1]`).

1. Run the testbed:

    ```bash
    python3 main.py --model-path onnx/alexnet_symbolic_shapes.onnx --strategy naive
    ```

## Possible research areas

- Design a tensor allocation strategy that is more optimal in terms of peak memory usage.
- Try to use more complex models: `ResNet`, `MobileNet` or maybe even transformers.
- Look into modern SoTA research papers, see MODeL.

## Links

- [ONNX](https://onnx.ai/)
- [netron.app: a visualizer for ONNX graphs](https://netron.app)
- [MODeL (Memory Optimizations for Deep Learning)](https://github.com/facebookresearch/MODel_opt)
