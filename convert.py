import argparse
import importlib
from modules.preprocess import prepare_cfg
from modules.model import load_model
import torch
import os
import subprocess
import modules.inception_next_nano

import warnings
warnings.filterwarnings("ignore")


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name')
    parser.add_argument('--stage', choices=["train_ce",'train_bce',])
    parser.add_argument('--openvino', action='store_true')
    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()
    model_name = args.model_name
    stage = args.stage

    cfg = importlib.import_module(f'configs.{model_name}').basic_cfg
    cfg = prepare_cfg(cfg,stage)

    model = load_model(cfg,stage, train=False)
    input_dummy = torch.randn(*cfg.input_shape)

    onxx_model_path = os.path.join(cfg.onnx_path, f"{model_name}.onnx")
    if not os.path.exists(cfg.onnx_path):
        os.makedirs(cfg.onnx_path)
    
    input_name = cfg.input_names[0]
    output_name = cfg.output_names[0]
    torch.onnx.export(
        model, input_dummy, onxx_model_path, verbose=True, input_names=cfg.input_names, output_names=cfg.output_names,opset_version=cfg.opset_version,
        dynamic_axes={input_name: {0 : 'batch_size'}, output_name: {0 : 'batch_size'}}
    )

    if args.openvino:
        import openvino as ov
        ov_model = ov.convert_model(onxx_model_path)

        out_path = os.path.join(cfg.openvino_path, f"{model_name}.xml")
        if not os.path.exists(cfg.openvino_path):
            os.makedirs(cfg.openvino_path)
        
        ov.save_model(ov_model, out_path, compress_to_fp16=True)
        print("openvino export completed.")

    return

if __name__=='__main__':
    main()