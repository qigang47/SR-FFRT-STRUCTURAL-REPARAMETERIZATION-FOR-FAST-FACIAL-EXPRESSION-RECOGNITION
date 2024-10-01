import onnx

onnx_model_path = '/Users/qigang/Desktop/FvfFER/syszux_scene.onnx'

# 加载 ONNX 模型
model = onnx.load(onnx_model_path)

# 打印所有节点的名称
for node in model.graph.node:
    print(node.name)
