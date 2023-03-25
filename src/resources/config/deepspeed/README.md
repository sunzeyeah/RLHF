# Deepspeed Configuration
- deepspeed==0.8.2
- transformers==4.26.1

- using ```transformers.Trainer``` and ```transformers.TrainingArguments```

Example of deepspeed config with key items explained:
```bash
{
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 100,
    "wall_clock_breakdown": false # "Enable timing of the latency of forward/backward/update training phases"
    
    "optimizer": {
        "type": "Adam",
        "params": {
          "lr": "auto",
          "betas": "auto",
          "eps": "auto",
          "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
    
    "fp16": {
        "enabled": "auto",
        "auto_cast": false, # automatically casts inputs to fp16
        "loss_scale": 0, # a fp16 parameter representing the loss scaling value for FP16 training. The default value of 0.0 results in dynamic loss scaling, otherwise the value will be used for static fixed loss scaling
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    # "BFLOAT16 requires hardware support (e.g., NVIDIA A100). Training with bfloat16 does not require loss scaling"
    "bf16": {
        "enabled": "auto"
    },
    
    "zero_optimization": {
        "stage": [0|1|2|3], # "Stage 0, 1, 2, and 3 refer to disabled, optimizer state partitioning, and optimizer+gradient state partitioning, and optimizer+gradient+parameter partitioning, respectively."
        "offload_optimizer": {
            "device": "[cpu|nvme]",
            "pin_memory": true, # "This feature can improve the throughput at the cost of making less memory available to other processes. Pinned memory is set aside to the specific process that requested it and its typically accessed much faster than normal CPU memory"
            # all nvme-related params
            "nvme_path": "/local_nvme",
            "buffer_count": 4,
            "fast_init": false
        },
        "offload_param": {
            "device": "[cpu|nvme]",
            "pin_memory": true, # "This feature can improve the throughput at the cost of making less memory available to other processes. Pinned memory is set aside to the specific process that requested it and its typically accessed much faster than normal CPU memory"
            # all nvme-related params
            "nvme_path": "/local_nvme",
            "buffer_count": 5,
            "buffer_size": 1e8,
            "max_in_cpu": 1e9
        },
        "overlap_comm": false, # "if set to true, trades off increased GPU RAM usage to lower all-reduce latency. overlap_comm uses 4.5x the allgather_bucket_size and reduce_bucket_size values. So if they are set to 5e8, this requires a 9GB footprint (5e8 x 2Bytes x 2 x 4.5). Therefore, if you have a GPU with 8GB or less RAM, to avoid getting OOM-errors you will need to reduce those parameters to about 2e8, which would require 3.6GB"
        "reduce_bucket_size": "auto", # "Number of elements reduced/allreduced at a time. Limits the memory required for the allgather for large model sizes. When set auto, it equals hidden_size*hidden_size"
        # only stage-2 params
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8, # "Number of elements allgathered at a time. Limits the memory required for the allgather for large model sizes"
        "reduce_scatter": true,
        "contiguous_gradients" : true, # "Copies the gradients to a contiguous buffer as they are produced. Avoids memory fragmentation during backward pass"
        "round_robin_gradients": [true|false], # "Stage 1 and 2 optimization for CPU offloading that parallelizes gradient copying to CPU memory among ranks by fine-grained gradient partitioning. Performance benefit grows with gradient accumulation steps (more copying between optimizer steps) or GPU count (increased parallelism)."
        # only stage-3 params
        "stage3_max_live_parameters" : 1e9, # "The maximum number of parameters resident per GPU before releasing. Smaller values use less memory, but perform more communication. 1e9 would consume ~2GB"
        "stage3_max_reuse_distance" : 1e9, # "Do not release a parameter if it will be reused within this threshold of parameters. Smaller values use less memory, but perform more communication."
        "stage3_prefetch_bucket_size" : "auto", # "The size of the fixed buffer for prefetching parameters. Smaller values use less memory, but can increase stalls due to communication. When set auto, it equals 0.9 * hidden_size * hidden_size"
        "stage3_param_persistence_threshold" : "auto", # "Do not partition parameters smaller than this threshold. Smaller values use less memory, but can greatly increase communication (especially latency-bound messages). When set auto, it equals 10 * hidden_size"
        "sub_group_size" : 1e12, # controls the granularity in which parameters are updated during optimizer steps. Parameters are grouped into buckets of sub_group_size and each buckets is updated one at a time. When used with NVMe offload in ZeRO-Infinity, sub_group_size therefore controls the granularity in which model states are moved in and out of CPU memory from NVMe during the optimizer step. This prevents running out of CPU memory for extremely large models. 
        "elastic_checkpoint" : [true|false],
        "stage3_gather_16bit_weights_on_model_save": true, # Consolidate the weights before saving the model by save_16bit_model(). Since the weights are partitioned across GPUs, they aren’t part of state_dict, so this function automatically gathers the weights when this option is enabled and then saves the fp16 model weights.
        "ignore_unused_parameters": true # Unused parameters in modules may be unexpected in static networks, but could be normal in dynamic networks. This controls whether or not training should terminate with an error message when unused parameters are detected
    },
  
    #  DeepSpeed Autotuner automatically discovers the optimal DeepSpeed configuration that delivers good training speed
    "autotuning": {},
  
    # Flops Profiler helps users easily measure both the model training/inference speed (latency, throughput) and efficiency (floating-point operations per second, i.e., FLOPS) of a model and its submodules
    "flops_profiler": {},
    
    "activation_checkpointing": {},
  
    "sparse_attention": {},
  
    # DeepSpeed Data Efficiency Library includes two techniques: curriculum learning and random layerwise token dropping (random-LTD).
    "data_efficiency": {},
  
    # Compression has seven different components, including layer reduction, weight quantization, activation quantization, sparse pruning, row pruning, head pruning, and channel pruning
    "compression_training": {}
}
```

Note that the speical value ```auto``` in the configuration will be automatically replaced with the correct or most efficient value from ```transformers.TrainingArguments```

## How to Choose Which ZeRO Stage and Offloads To Use For Best Performance

### Guideline

- Speed-wise (left is faster than right)

Stage 0 (DDP) > Stage 1 > Stage 2 > Stage 2 + offload > Stage 3 > Stage 3 + offloads

- GPU Memory usage-wise (right is more GPU memory efficient than left)

Stage 0 (DDP) < Stage 1 < Stage 2 < Stage 2 + offload < Stage 3 < Stage 3 + offloads

### Memory requirements
Since Deepspeed ZeRO can offload memory to CPU (and NVMe) the framework provides utils that allow one to tell how much CPU and GPU memory will be needed depending on the number of GPUs being used.

Using "bigscience/T0_3B" and one GPU as example:
```python
from transformers import AutoModel
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

model = AutoModel.from_pretrained("bigscience/T0_3B")
# stage 1 and 2
estimate_zero2_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)
# stage 3
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)
```

### Steps to find the optimal configuration
So when you want to get the fastest execution while fitting into minimal number of GPUs, here is the process you could follow. We start with the fastest approach and if running into GPU OOM we then go to the next slower approach, but which will use less GPU memory. And so on and so forth.

First of all set batch size to 1 (you can always use gradient accumulation for any desired effective batch size).

1. Enable ```--gradient_checkpointing 1``` (HF Trainer) or directly ```model.gradient_checkpointing_enable()``` - if OOM then

2. Try ZeRO stage 2 first. if OOM then

3. Try ZeRO stage 2 + ```offload_optimizer``` - if OOM then

4. Switch to ZeRO stage 3 - if OOM then

5. Enable ```offload_param``` to ```cpu``` - if OOM then

6. Enable ```offload_optimizer``` to ```cpu``` - if OOM then

7. If you still can’t fit a batch size of 1 first check various default values and lower them if you can. For example, if you use ```generate``` and you don’t use a wide search beam make it narrower as it’d take a lot of memory.

8. Definitely use mixed half-precision over fp32 - so bf16 on Ampere and higher GPUs and fp16 on older gpu architectures.

9. If you still OOM you could add more hardware or enable ZeRO-Infinity - that is switch offloads ```offload_param``` and ```offload_optimizer``` to ```nvme```. You need to make sure it’s a very fast nvme.

You can, of course, work through these steps in reverse by starting with the most GPU memory efficient config and then going backwards. Or try bi-secting it.


## Tricks & Troubleshooting
- If you are training from scratch, try to have tensors with shapes that are divisible by 16 (e.g. hidden size). For batch size try divisible by 2 at least. There are wave and tile quanitization divisibility that is hardware-specific if you want to squeeze even higher performance from your GPUs.

- It’s possible to adjust ZeRO-3 configuration to make it perform closer to ZeRO-2:
    - set ```stage3_param_persistence_threshold``` to a very large number - larger than the largest parameter, e.g., ```6 * hidden_size * hidden_size```. This will keep the parameters on the GPUs. 
    - turn off ```offload_params``` since ZeRO-2 doesn’t have that option.

- ```overlap_comm```: if true, trades off increased GPU RAM usage to lower all-reduce latency. ```overlap_comm``` uses 4.5x the ```allgather_bucket_size``` and ```reduce_bucket_size``` values. So if they are set to 5e8, this requires a 9GB footprint (5e8 x 2Bytes x 2 x 4.5). Therefore, if you have a GPU with 8GB or less RAM, to avoid getting OOM-errors you will need to reduce those parameters to about 2e8, which would require 3.6GB. You will want to do the same on larger capacity GPU as well, if you’re starting to hit OOM

- ```sub_group_size```: You can leave it to default value when not using NVMe offload. You may want to change its default value in the following cases: 
  - Running into OOM during optimizer step: Reduce sub_group_size to reduce memory utilization of temporary buffers; 
  - Optimizer Step is taking a long time: Increase sub_group_size to improve bandwidth utilization as a result of the increased data buffers.

- ```activation_checkpointing```: activation checkpointing and gradient checkpointing refer to the same methodology. But enabling ```activation_checkpointing``` in deepSpeed config has no effect on huggingface transformers. If you want to use a HF Transformers models you can do model.gradient_checkpointing_enable() or use --gradient_checkpointing in the HF Trainer, which will automatically enable this for you.

- Using fp16 and you see in your log that Deepspeed reports OVERFLOW! as follows:
```bash
[deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 262144, reducing to 262144
```
that means that the Deepspeed loss scaler can’t figure out a scaling co-efficient that overcomes loss overflow. In this case you usually need to raise the value of ```initial_scale_power``` to 32 which will typically solve the problem.


# Main Resources

- [Huggingface Deepspeed Integration](https://huggingface.co/docs/transformers/main_classes/deepspeed#deepspeed-integration)
- [Deepspeed Docs](https://www.deepspeed.ai/)
- [Deepspeed Github](https://github.com/microsoft/DeepSpeed)