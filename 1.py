from onnx_coreml import convert

# 这里假设你已经有了一个ONNX模型的路径
onnx_model_path = '/Users/qigang/Desktop/FvfFER/syszux_scene.onnx'


# 转换ONNX模型为CoreML模型
def convert_onnx_to_coreml(onnx_model_path):

    # 将ONNX模型转换为CoreML模型
    try:
        coreml_model = convert(
            model=onnx_model_path,
        )
        return coreml_model
    except KeyError as e:
        print("A KeyError occurred: ", e)
        # 这里处理KeyError，你可以决定如何处理它。
        # 比如你可以记录下来错误并继续转换，但是请记住，
        # 这可能会导致生成的CoreML模型有缺陷。
        pass


# 调用函数并获取CoreML模型
coreml_model = convert_onnx_to_coreml(onnx_model_path)

# 如果需要保存CoreML模型
coreml_model.save('your_model.mlmodel')
