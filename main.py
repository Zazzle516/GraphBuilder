# -*- coding: utf-8 -*-
import os
import argparse
import logging
import torch
import re
import json
from torch_mlir.ir import Module
from safetensors.torch import load_file
from graph_builder import *
from config import model2config
from layers import *

for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

logging.basicConfig(
    filename="./my_model_config.log",   
    filemode="w",                      
    level=logging.INFO,                 
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',
                        type=str,
                        default='mhlo-model',
                        help='The path to save the mhlo ir file')

    args = parser.parse_args()
    return args

def compile_hf_to_middle_ir(model_dir: str, out_dir = None,dtype = torch.float16,logger=None) -> Module:
    """
    parse hf-model then hard-coding model layers
    """
    assert model_dir is not None
    with open(f"{model_dir}/config.json",'r',encoding='utf-8') as f:
        hf_config = json.load(f)
    model_class = "qwen"    # Fixed for Test
    # type_match = re.search(r'[^A-Za-z]', hf_config["model_type"])
    # if type_match:
    #     idx = type_match.start()
    #     model_class = hf_config["model_type"][:idx]
    # else:
    #     model_class = hf_config["model_type"]       

    if not model_class in model2config.keys():
        raise IndexError("model type not support,could not find config")
    config = model2config[model_class].canonicalize(hf_config, dtype)

    weight_path = os.path.join(model_dir, 'model.safetensors')
    if not os.path.isfile(weight_path):
        return None
    weights = load_file(weight_path)

    # 目前支持建图的模型类型
    if not model_class in model2builder.keys():
        raise IndexError("model type not support,could not find graphbuilder")
    # 直接调用对应模型的 init 函数   注意这里的调用层次
    modelBuilder = model2builder[model_class](weights,config)

    modelBuilder.build(weights)
    modelBuilder.convert_to_mhlo("qwen3-0.6B")
    if logger is not None:
        logger.info(weights)
        logger.info(modelBuilder.importer.print_module())
    return modelBuilder.importer.get_module()


def main():
    print("prepare to transform hf-model to mhlo ir for qwen3-0.6B")
    print("log file path:", os.path.abspath("my_model_config.log"))
   
    model_dir = "/home/zazzle/Models/Qwen3-0.6B"
    if model_dir is None:
        raise EnvironmentError (
            "The enviroment variable of hf-model is not been set yet"
        )
    
    args = parse_arguments()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    compile_hf_to_middle_ir(model_dir,logger=logger)
    print("success")
    
if __name__ == '__main__':
    #export QWEN3_0_6B=${PWD}/qwen3_0.6B
    main()
