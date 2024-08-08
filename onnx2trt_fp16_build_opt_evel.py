import numpy as np
import os
import tensorrt as trt
opt_level = 0

def onnx2trt(onnxFile, plan_name, min_shapes, opt_shapes, max_shapes, max_workspace_size = None, use_fp16=False, builder_opt_evel=None):
    logger = trt.Logger(trt.Logger.VERBOSE)

    # H 创建builder
    builder = trt.Builder(logger)

    # H 声明一个builder config对象
    config = builder.create_builder_config()

    # H 初始化config对象
    if max_workspace_size:
        config.max_workspace_size = max_workspace_size
    else:
        config.max_workspace_size = 4<<30  # 10GB 最大工作空间大小

    # H 声明一个builder network对象
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # H 根据config对象和network对象构建解析对象parser
    parser = trt.OnnxParser(network, logger)

    if not os.path.exists(onnxFile):
        print("Failed finding onnx file!")
        exit()
    print("Succeeded finding onnx file!")

    # H 使用parser解析ONNX模型文件
    with open(onnxFile, 'rb') as model:
        # import pdb; pdb.set_trace()
        (onnx_path, _) = os.path.split(onnxFile)

        if not parser.parse(model.read(), path=onnxFile):
            print("Failed parsing ONNX file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
    print("Succeeded parsing ONNX file!")

    # H 进一步的 初始化config对象
    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if builder_opt_evel:
        config.builder_optimization_level = builder_opt_evel   # TensorRT在构建推理引擎时所使用的优化级别

    # set profile
    assert network.num_inputs == len(min_shapes)
    assert network.num_inputs == len(opt_shapes)
    assert network.num_inputs == len(max_shapes)

    # H 声明一个builder profile对象，用于B对象的更改
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        profile.set_shape(input_tensor.name, tuple(min_shapes[i]), tuple(opt_shapes[i]), tuple(max_shapes[i]))

    # H config对象接受profile对象
    config.add_optimization_profile(profile)

    # H 根据config对象和network对象构建engine
    engine = builder.build_engine(network, config)
    if not engine:
        raise RuntimeError("build_engine failed")
    print("Succeeded building engine!")

    # H engine对象序列化
    print("Serializing Engine...")
    serialized_engine = engine.serialize()
    if serialized_engine is None:
        raise RuntimeError("serialize failed")

    (plan_path, _) = os.path.split(plan_name)
    os.makedirs(plan_path, exist_ok=True)
    with open(plan_name, "wb") as fout:
        fout.write(serialized_engine)

def export_clip_model():
    onnx_path = "./onnx/CLIP.onnx"
    plan_path = "./engine_fp16_opt_level/CLIP_{}.plan".format(opt_level)

    # 1. 统一min_shape\opt_shape\max_shape为77; 2. 使用FP16
    onnx2trt(onnx_path, plan_path, [(1, 77)], [(1, 77)], [(1, 77)], use_fp16=True,builder_opt_evel=opt_level)

    print("======================= CLIP onnx2trt done!")

def export_control_net_model():
    def get_shapes(B, S):
        return [(B, 4, 32, 48), (B, 3, 256, 384), tuple([B]), (B, S, 768)]

    onnx_path = "./onnx/ControlNet.onnx"
    plan_path = "./engine_fp16_opt_level/ControlNet_{}.plan".format(opt_level)

    # 1. 统一min_shape\opt_shape\max_shape为77; 2. 使用FP16
    onnx2trt(onnx_path, plan_path,
             get_shapes(1, 77),
             get_shapes(1, 77),
             get_shapes(1, 77),
             use_fp16=True, builder_opt_evel=opt_level)

    print("======================= ControlNet onnx2trt done!")

def export_controlled_unet_model():
    def get_shapes(B, S):
        return [(B, 4, 32, 48), tuple([B]), (B, S, 768),
                (B, 320, 32, 48),
                (B, 320, 32, 48),
                (B, 320, 32, 48),
                (B, 320, 16, 24),
                (B, 640, 16, 24),
                (B, 640, 16, 24),
                (B, 640, 8, 12),
                (B, 1280, 8, 12),
                (B, 1280, 8, 12),
                (B, 1280, 4, 6),
                (B, 1280, 4, 6),
                (B, 1280, 4, 6),
                (B, 1280, 4, 6)]

    onnx_path = "./onnx/ControlledUnet"
    onnx_path = onnx_path + "/ControlledUnet.onnx"

    plan_path = "./engine_fp16_opt_level/ControlledUnet_{}.plan".format(opt_level)

    # 1. 统一min_shape\opt_shape\max_shape为77; 2. 使用FP16
    onnx2trt(onnx_path, plan_path,
             get_shapes(1, 77),
             get_shapes(1, 77),
             get_shapes(1, 77),
             use_fp16=True,builder_opt_evel=opt_level)

    print("======================= ControlNet onnx2trt done!")

def export_decoder_model():
    onnx_path = "./onnx/Decoder.onnx"
    plan_path = "./engine_fp16_opt_level/Decoder_{}.plan".format(opt_level)

    # 1. 统一min_shape\opt_shape\max_shape; 2. 使用FP16
    onnx2trt(onnx_path, plan_path, [(1, 4, 32, 48)], [(1, 4, 32, 48)], [(1, 4, 32, 48)], use_fp16=True,builder_opt_evel=opt_level)

    print("======================= Decoder  onnx2trt done!")




def main():
    export_clip_model()
    export_control_net_model()
    export_controlled_unet_model()
    export_decoder_model()

if __name__ == '__main__':
    main()
