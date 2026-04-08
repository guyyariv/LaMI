# LaMI: Augmenting Large Language Models via Late Multi-Image Fusion
This repo contains the official PyTorch implementation of [*LaMI: Augmenting Large Language Models via Late Multi-Image Fusion*](https://arxiv.org/abs/2406.13621v2) (ACL 2026).

# Abstract
Commonsense reasoning often requires both textual and visual knowledge, yet Large Language Models (LLMs) trained solely on text lack visual grounding (e.g., "what color is an emperor penguin's belly?"). Visual Language Models (VLMs) perform better on visually grounded tasks but face two limitations: (i) often reduced performance on text-only commonsense reasoning compared to text-trained LLMs, and (ii) adapting newly released LLMs to vision input typically requires costly multimodal training. An alternative augments LLMs with test-time visual signals, improving visual commonsense without harming textual reasoning, but prior designs often rely on early fusion and a single image, which can be suboptimal. We propose a late multi-image fusion method: multiple images are generated from the text prompt with a lightweight parallel sampling, and their prediction probabilities are combined with those of a text-only LLM through a late-fusion layer that integrates projected visual features just before the final prediction. Across visual commonsense and NLP benchmarks, our method significantly outperforms augmented LLMs on visual reasoning, matches VLMs on vision-based tasks, and, when applied to strong LLMs such as LLaMA 3, also improves NLP performance while adding only modest test-time overhead.

<a href="https://arxiv.org/abs/2406.13621v2"><img src="https://img.shields.io/badge/arXiv-2406.13621-b31b1b.svg" height=22.5></a>
<a href="https://pages.cs.huji.ac.il/adiyoss-lab/vLMIG/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=22.5></a>
<a href="https://colab.research.google.com/drive/1-idBJHvI9cPAQ7GQq5in-4Sa_wiHzQkT?usp=sharing"><img src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252" height=22.5></a>

![figure3](https://github.com/guyyariv/vLMIG/assets/89798559/e2a46b40-b7ca-4fea-80dd-ee15d1fa18f4)


# Installation
```
git clone git@github.com:guyyariv/LaMI.git
cd LaMI
python -m venv lami
source lami/bin/activate
pip install -r requirements.txt
```

# Pre-Trained Models
Download the pre-trained models provided in the paper using the following commands:

First, run ```pip install gdown```

#### GPT-2
```angular2html
mkdir -p output/gpt2 && \
gdown "https://drive.google.com/uc?id=1ZvJTXiuXjcCCwcm_PeQ78hxZCEab4tID" -O output/gpt2/ft_wiki_laion_220_2.bin
```

#### Gemma-2B
```angular2html
mkdir -p output/gemma_2b && \
gdown "https://drive.google.com/uc?id=14qWLXJNMcOmgMVkQa7KAdKbcucz97_o8" -O output/gemma_2b/ft_wiki_laion_220_2.bin
```

#### LLaMA 3
```angular2html
mkdir -p output/llama3 && \
gdown "https://drive.google.com/uc?id=14qWLXJNMcOmgMVkQa7KAdKbcucz97_o8" -O output/llama3/ft_wiki_laion_220_2.bin
```

# Training

Configure your Accelerate environment with:
```angular2html
accelerate config
```

Launch the training process:
```angular2html
accelerate launch train.py \
--run_name test \
--dataset_name cropped_vg \
--model_name_or_path meta-llama/Meta-Llama-3-8B \
--output_dir output/llama3 \
--report_to wandb \
--per_device_train_batch_size 64 \
--num_train_epochs 1 \
--run_bf16 \
--learning_rate 5e-4 \
--with_tracking
```
For full reproducibility, ensure to fine-tune your trained model on Wikipedia-103 (max_elements=200,000) and LAION-220.

# Evaluation

We evaluate the model on multiple benchmarks:

#### Visual Commonsense:
For ImageNetVC evaluation (based on the official implementation https://github.com/hemingkx/ImageNetVC/blob/main/VaLM/BLIP-2/ImageNetVC.py):
```angular2html
python3 eval_scripts/imagenetVC.py --model_name meta-llama/Meta-Llama-3-8B --run_name llama3_imagenetvc --pretrained_model output/llama3/ft_wiki_laion_220_2 --generate_images True --k 10
```
Access script parameters with:
```angular2html
python3 eval_scripts/imagenetVC.py --help
```

#### Commonsense:
For commonsense evaluation:
```angular2html
python3 eval_scripts/commonsenseQA.py --model_name meta-llama/Meta-Llama-3-8B --pretrained_model output/llama3/ft_wiki_laion_220_2 --generate_images True --k 10 --testset piqa
```
Note: This command example runs the testset PIQA, but this script can also be used to evaluate other datasets, such as SIQA, ARC, etc., by choosing ``` --testset {testset name} ```.
Access script parameters with:
```angular2html
python3 eval_scripts/commonsenseQA.py --help
```

#### Reading Comprehension
For SQuAD, QUAC, and BoolQ run respectively:
```angular2html
python3 eval_scripts/squad.py --model_name meta-llama/Meta-Llama-3-8B --pretrained_model output/llama3/ft_wiki_laion_220_2 --generate_images True --k 10
python3 eval_scripts/quac.py --model_name meta-llama/Meta-Llama-3-8B --pretrained_model output/llama3/ft_wiki_laion_220_2 --generate_images True --k 10
python3 eval_scripts/boolq.py --model_name meta-llama/Meta-Llama-3-8B --pretrained_model output/llama3/ft_wiki_laion_220_2 --generate_images True --k 10
```
Access script parameters with:
```angular2html
python3 eval_scripts/squad.py --help
python3 eval_scripts/quac.py --help
python3 eval_scripts/boolq.py --help
```

# Acknowledgments
Our code is partially built upon [Transformers training example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling) and [ImagenetVC](https://github.com/hemingkx/ImageNetVC/tree/main).

# Cite
If you use our work in your research, please cite the following paper:
```
@inproceedings{yariv2026lami,
  title={LaMI: Augmenting Large Language Models via Late Multi-Image Fusion},
  author={Yariv, Guy and Schwartz, Idan and Adi, Yossi and Benaim, Sagie},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026}
}
```

# License
This repository is released under the MIT license as found in the [LICENSE](LICENSE) file.
