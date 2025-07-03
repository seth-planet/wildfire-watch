#!/usr/bin/env python3
"""Check ONNX model input and output node names."""

import sys
import onnx

if len(sys.argv) < 2:
    print("Usage: python3 check_onnx_nodes.py <model.onnx>")
    sys.exit(1)

model_path = sys.argv[1]
model = onnx.load(model_path)

print(f"\nModel: {model_path}")
print("\nInputs:")
for i, input in enumerate(model.graph.input):
    print(f"  [{i}] {input.name} - shape: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}")

print("\nOutputs:")
for i, output in enumerate(model.graph.output):
    print(f"  [{i}] {output.name} - shape: {[dim.dim_value for dim in output.type.tensor_type.shape.dim]}")

# Also check all node names
print("\nAll nodes (first 10):")
for i, node in enumerate(model.graph.node[:10]):
    print(f"  [{i}] {node.op_type}: {node.name}")
    if node.output:
        print(f"       outputs: {node.output}")