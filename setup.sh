#!/bin/bash

python3 onnx/download_models.py

for model_path in onnx/*.onnx; do
    model_name=$(basename "$model_path" .onnx)
    if [[ "$model_path" == *"_symbolic_shapes.onnx" ]] || [ -f "onnx/${model_name}_symbolic_shapes.onnx" ]; 
    then
        continue
    fi
    echo "processing model: $model_name"
    output_path="onnx/${model_name}_symbolic_shapes.onnx"
    python3 tools/symbolic_shape_infer.py --input "$model_path" --output "$output_path"
    echo "processing is finished, saved to the: $output_path"
done

echo "done"
