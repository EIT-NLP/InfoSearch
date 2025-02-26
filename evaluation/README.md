<h1 align="center">InfoSearch--Beyond Content Relevance: Evaluating Instruction Following in Retrieval Models</b></h1>

<h4 align="center">
    <p>
        <a href="#links">Model/Data Links</a> |
        <a href="#installation">Installation</a> |
        <a href="#usage">Usage</a> |
        <a href="#citing">Citing</a> |
    <p>
</h4>

Official repository for the paper [Beyond Content Relevance: Evaluating Instruction Following in Retrieval Models]().

## Links
- The evaluation code leverages the scripts provided by the [FollowIR](https://github.com/orionw/FollowIR) framework, which offer robust tools for assessing instruction-following capabilities.
### Test Datasets

| Dimension                                                           | Description                                                                    |
|:--------------------------------------------------------------------|:-------------------------------------------------------------------------------|
| [Language-v1](https://huggingface.co/datasets/jianqunZ/Language-v1) | Including the conditions [Chinese], [English]                                  |
| [Clarity-v1](https://huggingface.co/datasets/jianqunZ/Clarity-v1)   | Including the condition [keyword] which need to be exactly matched in document |
| [Audience-v1](https://huggingface.co/datasets/jianqunZ/Audience-v1) | Including the conditions [layman], [expert]                                    |
| [Length-v1](https://huggingface.co/datasets/jianqunZ/Length-v1)     | Including the conditions [sentence], [paragraph], [article]                    |
| [Source-v1](https://huggingface.co/datasets/jianqunZ/Source-v1)     | Including the conditions [blog], [forum post], [news]                          |
| [Format-v1](https://huggingface.co/datasets/jianqunZ/Format-v1)     | Including the conditions [post], [code], [manual]                              |

### Train Dataset

| Dataset              | Description                                   |
|:---------------------|:----------------------------------------------|
| [InfoSearch-train]() | We design a train dataset to fine-tune models |

### Lise-wise Rerank


### Model Fine-tuning

We use `LLaMa-Factory` to fine-tune FollowIR-7B to create InfoSearch-7B , after transforming it to fit their format (
input of "query" + "instruction" inside the template, output is the label, and instruction as the beginning of the
template) with the following training script:

```bash
#!/bin/bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path "jhu-clsp/FollowIR-7B" \
    --dataset InfoSearch-train \
    --template mistral \
    --output_dir InfoSearch-finetune \
    --finetuning_type lora \
    --lora_target q_proj,v_proj,o_proj,k_proj \
    --overwrite_cache \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 2 \
    --save_steps 25 \
    --learning_rate 3e-5 \
    --num_train_epochs 8.0 \
    --max_length 2048 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --plot_loss \
    --bf16
```

## Installation

If you wish to reproduce the experiments in the paper you can use the following code:

```bash
git clone https://github.com/EIT-NLP/InfoSearch.git
cd InfoSearch/evaluation/
conda create -n infosearch python=3.9 -y
conda activate infosearch
pip install -r requirements.txt
```

We modified the `mteb` library to calculate the **WISE** and **SICR** metrics. You can install the modified version of
the library by running the following command:

```bash
cd mteb_infosearch
pip install mteb_infosearch-0.1.0-py3-none-any.whl --no-deps
```

## Usage

Our task names include:

- Language-v1
- Clarity-v1
- Audience-v1
- Length-v1
- Source-v1
- Format-v1

Here is an example of how to evaluate a model on a specified task:

```bash
cd evaluation/

CUDA_VISIBLE_DEVICES=0 python models/e5/evaluate_e5.py \
--model_name_or_path infloat/e5-large-v2 \
--output_dir e5-large_result \
--batch_size 32 \
--pool_type avg \
--task_names Language-v1
```

You can also evaluate on multiple tasks by adding more task names to the `--task_names` argument. Each task should be
separated by a comma.

```bash
--task_names Language-v1, Keyword-v1, Audience-v1
```

### Point-wise Reranker Usage

If you want to evaluate with your own reranker model, you need to add your model to MODEL_DICT in `reranker_models.py`
and then run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python models/rerankers/evaluate_reranker.py \
--model_name_or_path /path/to/your/model \
--output_dir /path/to/output/dir \
--batch_size 16 \
--task_names Language-v1
```

It will take a while to evaluate a model on each task.

### List-wise Reranker Usage
[RankLLM](https://github.com/castorini/rank_llm) provides useful tools for list-wise reranking.

```bash
git clone https://github.com/castorini/rank_llm.git
cd rank_llm
conda create -n rankllm python=3.10 -y
conda activate rankllm
pip install -r requirements.txt
```

We provide a script to rerank retrieval results and retrieval results based on `e5-mistral` under the `listwise` directory. 
If you want to rerank retrieval results, you need to put the retrieval results in the `rank_llm/src/rankllm/dimension` directory.
and put the `rerank.py` script in the `rank_llm/src/rankllm` directory.

```plaintext
rank_llm/
├── src/
│   ├── rankllm/
│   │   ├── ...
│   │   ├── dimension
│   │   │   ├── Audience-v1_ch_retrieval_results.json
│   │   │   ├── ...
│   │   ├── ...
│   │   ├── rerank.py
│   │   ├── ...
├── ...
```

The retrieval results should be structured as follows:

```json
[
  {
    "query": {
      "text": "Do Walnuts Really Improve Artery Function?",
      "qid": "0"
    },
    "candidates": [
      {
        "docid": "0",
        "score": 1.0,
        "doc": {
          "content": "Walnuts, like other nuts, have been found to have potential benefits for..."
        }
      },
      {
        "docid": "1",
        "score": 0.5,
        "doc": {
          "content": "The consumption of nuts, such as almonds and pistachios, has been linked to numerous health benefits..."
        }
      },
      ...
    ]
  },
  ...
]
```

```bash
# Make sure the current directory is rank_llm/src
CUDA_VISIBLE_DEVICES=0 python -m rank_llm.rerank.py \
--model_path /path/to/your/model \
--dataset Language-v1 \
--context_size 4096 \
--top_k_candidates 100
```
## Citing

If you found the code, data or model useful, free to cite:
```bibtex
@article{zhou2024beyond,
  title={Beyond Content Relevance: Evaluating Instruction Following in Retrieval Models},
  author={Zhou, Jianqun and Zheng, Yuanlei and Chen, Wei and Zheng, Qianqian and Shang, Zeyuan and Zhang, Wei and Meng, Rui and Shen, Xiaoyu},
  journal={arXiv preprint arXiv:2410.23841},
  year={2024}
}
```

