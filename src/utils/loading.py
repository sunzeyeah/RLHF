import os
import torch
import glob
from typing import Callable, Dict, Tuple

# Register load pipelines via module import
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.models.llama import LlamaModel
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from accelerate.utils import get_balanced_memory
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)

from src.utils.file_utils import print_trainable_parameters
from src.data.pipeline import _DATAPIPELINE
from src.models.trainer import _TRAINERS, register_trainer
from src.models.llama import _prepare_decoder_attention_mask
# from trlx.pipeline.offline_pipeline import PromptPipeline
# from trlx.trainer.accelerate_ilql_trainer import AccelerateILQLTrainer
# from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer
# from trlx.trainer.accelerate_sft_trainer import AccelerateSFTTrainer

try:
    from src.models.trainer import NeMoILQLTrainer
except ImportError:
    # NeMo is not installed
    def _trainer_unavailble(name):
        def log_error(*args, **kwargs):
            raise ImportError(f"Unable to import NeMo so {name} is unavailable")

        return register_trainer(name)(log_error)

    _trainer_unavailble("NeMoILQLTrainer")


def prepare_decoder_attention_mask(self, **kwargs):
    return _prepare_decoder_attention_mask(**kwargs)


def chatglm_auto_configure_device_map(num_gpus: int, model_name: str, local_rank: int = 0) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_hidden_layers = 28
    layers_per_gpu = (num_hidden_layers+2) // num_gpus
    layer_prefix = 'transformer'

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上

    encode = ""
    if 'chatglm2' in model_name:
        device_map = {
            f"{layer_prefix}.embedding.word_embeddings": local_rank,
            f"{layer_prefix}.rotary_pos_emb": local_rank,
            f"{layer_prefix}.output_layer": local_rank,
            f"{layer_prefix}.encoder.final_layernorm": local_rank,
            f"base_model.model.output_layer": local_rank,
        }
        encode = ".encoder"
    else:
        device_map = {
            f'{layer_prefix}.word_embeddings': local_rank,
            f'{layer_prefix}.final_layernorm': local_rank,
            'lm_head': local_rank,
            f'base_model.model.lm_head': local_rank,
        }
    used = 2
    gpu_target = 0
    # TODO: Assuming CUDA device index is consecutive, e.g. cuda:0, cuda:1, cuda:2
    for i in range(num_hidden_layers):
        if used >= layers_per_gpu + (gpu_target % 2):
            gpu_target += 1
            gpu_target %= num_gpus
            used = 0
        device_map[f'{layer_prefix}{encode}.layers.{i}'] = gpu_target + local_rank
        used += 1

    return device_map


def llama_and_baichuan_auto_configure_device_map(num_gpus: int, model_name: str, local_rank: int = 0) -> Dict[str, int]:
    layer_prefix = 'model'
    # model.embed_tokens 占用1层
    # model.norm 和 lm_head 占用1层
    # model.layers 占用 num_hidden_layers 层
    # 总共num_hidden_layers+2层分配到num_gpus张卡上
    if "7b" in model_name.lower():
        num_hidden_layers = 32
    elif "13b" in model_name.lower():
        num_hidden_layers = 40
    else:
        raise ValueError(f"Only supports baichuan-7B, baichuan-13B, llama-7B and llama-13B, but {model_name} is provided")

    layers_per_gpu = (num_hidden_layers+2) // num_gpus
    device_map = {
        f'{layer_prefix}.embed_tokens':  local_rank,
        f'{layer_prefix}.norm': local_rank,
        'lm_head': local_rank,
        f'base_model.model.lm_head': local_rank,
    }
    used = 2
    gpu_target = 0
    # TODO: Assuming CUDA device index is consecutive, e.g. cuda:0, cuda:1, cuda:2
    for i in range(num_hidden_layers):
        if used >= layers_per_gpu + (gpu_target % 2):
            gpu_target += 1
            gpu_target %= num_gpus
            used = 0
        device_map[f'{layer_prefix}.layers.{i}'] = gpu_target + local_rank
        used += 1

    return device_map


def load_params_8bit_or_4bit(args, model: PreTrainedModel) -> Dict:
    # init bnb config for quantization
    bf16 = torch.cuda.get_device_capability()[0] >= 8
    if bf16:
        bnb_4bit_compute_dtype = torch.bfloat16
    else:
        bnb_4bit_compute_dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=args.bits == 8,
        load_in_4bit=args.bits == 4,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype
    )
    params = {
        "low_cpu_mem_usage": True,
        'quantization_config': bnb_config
    }
    # infer device map
    if args.multi_card:
        max_memory = get_balanced_memory(model, dtype=torch.int8, low_zero=False,
                                         no_split_module_classes=model._no_split_modules)
        params['device_map'] = infer_auto_device_map(
            model,
            dtype=torch.int8,
            no_split_module_classes=model._no_split_modules,
            max_memory=max_memory
        )
    else:
        params['device_map'] = {"": args.local_rank}

    return params


def load_tokenizer_and_model(args, with_trainer: bool = True) -> Tuple[PreTrainedTokenizer, PreTrainedModel, int]:
    # load tokenizer
    tokenizer_path = args.tokenizer_path if hasattr(args, "tokenizer_path") else args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # set eos token
    if "chatglm2" in args.model_name_or_path.lower():
        eos_token_id = tokenizer.get_command("eop") if args.checkpoint is not None else tokenizer.get_command("<eos>")
    elif "chatglm1_1" in args.model_name_or_path.lower():
        eos_token_id = tokenizer.eos_token_id
    elif "chatglm" in args.model_name_or_path.lower():
        eos_token_id = tokenizer.eop_token_id
    elif "baichuan" in args.model_name_or_path.lower():
        eos_token_id = tokenizer.bos_token_id if args.checkpoint is not None else tokenizer.eos_token_id
    else:
        eos_token_id = tokenizer.eos_token_id

    # load model
    if "chatglm" in args.model_name_or_path.lower():
        model_class = AutoModelForSeq2SeqLM
    else:
        model_class = AutoModelForCausalLM

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    params = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "load_in_8bit": hasattr(args, "bits") and args.bits == 8,
        "load_in_4bit": hasattr(args, "bits") and args.bits == 4,
        # "quantization_config": bnb_config,
    }
    if with_trainer:
        params["device_map"] = args.device_map
        params["low_cpu_mem_usage"] = True
    model = model_class.from_pretrained(args.model_name_or_path,
                                        **params)
    # # cpu
    # if not torch.cuda.is_available():
    #     model = model_class.from_pretrained(args.model_name_or_path,
    #                                         trust_remote_code=True)
    # # 8bit or 4bit
    # elif hasattr(args, "bits") and args.bits in [4, 8]:
    #     config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    #     model = model_class.from_config(config, trust_remote_code=True)
    #     params = load_params_8bit_or_4bit(args, model)
    #     model = model_class.from_pretrained(args.model_name_or_path,
    #                                         trust_remote_code=True,
    #                                         **params)
    #     if args.do_train:
    #         if args.gradient_checkpointing:
    #             model.gradient_checkpointing_enable()
    #         model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    # # multi gpu card
    # elif hasattr(args, "multi_card") and args.multi_card:
    #     with init_empty_weights():
    #         config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    #         model = model_class.from_config(config, trust_remote_code=True).half()
    #     model.tie_weights()
    #     if "llama" in args.model_name_or_path.lower() or \
    #         "baichuan" in args.model_name_or_path.lower() or \
    #         "vicuna" in args.model_name_or_path.lower():
    #         device_map = llama_and_baichuan_auto_configure_device_map(
    #             torch.cuda.device_count(),
    #             args.model_name_or_path.lower(),
    #             args.local_rank
    #         )
    #     elif "chatglm" in args.model_name_or_path.lower():
    #         device_map = chatglm_auto_configure_device_map(
    #             torch.cuda.device_count(),
    #             args.model_name_or_path.lower(),
    #             args.local_rank
    #         )
    #     else:
    #         #     max_memory = get_balanced_memory(model, dtype=torch.float16, low_zero=False,
    #         #                                      no_split_module_classes=model._no_split_modules)
    #         #     device_map = infer_auto_device_map(model, dtype=torch.float16, max_memory=max_memory,
    #         #                                        no_split_module_classes=model._no_split_modules)
    #         device_map = "auto"
    #
    #     model = load_checkpoint_and_dispatch(model,
    #                                          checkpoint=args.model_name_or_path,
    #                                          device_map=device_map,
    #                                          no_split_module_classes=model._no_split_modules,
    #                                          dtype=torch.float16)
    # # single gpu card
    # else:
    #     model = model_class.from_pretrained(args.model_name_or_path,
    #                                         trust_remote_code=True,
    #                                         torch_dtype=torch.float16,
    #                                         device_map={"": args.local_rank})

    # post-loading operations
    if hasattr(args, "concat_samples") and args.concat_samples:
        funcType = type(LlamaModel._prepare_decoder_attention_mask)
        model.model._prepare_decoder_attention_mask = funcType(prepare_decoder_attention_mask, model.model, LlamaModel)
    if "pangu" in args.model_name_or_path.lower():
        model.resize_token_embeddings(tokenizer.vocab_size)
    if hasattr(args, "bits") and args.bits in [4, 8] and args.do_train:
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    # init peft model (if necessary)
    if hasattr(args, "lora_rank") and args.lora_rank > 0:
        model = get_peft_model(args, model)

    return tokenizer, model, eos_token_id


def get_peft_model(args, model: PreTrainedModel) -> PreTrainedModel:
    if "llama" in args.model_name_or_path.lower() or \
        "vicuna" in args.model_name_or_path.lower() or \
        "billa" in args.model_name_or_path.lower() or \
        "atomgpt" in args.model_name_or_path.lower() or \
        "pangu" in args.model_name_or_path.lower():
        target_modules = "q_proj,k_proj,v_proj"
        task_type = "CAUSAL_LM"
    elif "baichuan" in args.model_name_or_path.lower():
        target_modules = "W_pack"
        task_type = "CAUSAL_LM"
    elif "bloom" in args.model_name_or_path.lower() or "tigerbot" in args.model_name_or_path.lower():
        target_modules = "query_key_value"
        task_type = "CAUSAL_LM"
    elif "glm" in args.model_name_or_path.lower():
        target_modules = "query_key_value"
        task_type = "SEQ_2_SEQ_LM"
    else:
        raise ValueError(f"Unsupported model name: {args.model_name_or_path}")

    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias=args.lora_train_bias,
        task_type=task_type
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    return model


def load_checkpoint(args, model: PreTrainedModel, strict: bool = True) -> None:
    checkpoints = glob.glob(args.checkpoint.replace("star", "*"))
    st = dict()
    for checkpoint in checkpoints:
        st.update(torch.load(checkpoint, map_location="cpu"))
    model.load_state_dict(st, strict=strict)
    del st


def get_trainer(name: str) -> Callable:
    """
    Return constructor for specified RL model trainer
    """
    name = name.lower()
    if name in _TRAINERS:
        return _TRAINERS[name]
    else:
        raise Exception("Error: Trying to access a trainer that has not been registered")


def get_pipeline(name: str) -> Callable:
    """
    Return constructor for specified pipeline
    """
    name = name.lower()
    if name in _DATAPIPELINE:
        return _DATAPIPELINE[name]
    else:
        raise Exception("Error: Trying to access a pipeline that has not been registered")
