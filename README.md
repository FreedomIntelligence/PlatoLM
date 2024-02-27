# PlatoLM: Teaching LLMs via a Socratic Questioning User Simulator
# ✨ Latest News
- [01/16/2024]: We are rejected by [ICLR-2024](https://openreview.net/forum?id=9nddtu94uX) with scores 8666（ranked top 13%-16%） due to AC's Factual Error but no one replies the appeal email. Fine.
- [10/12/2023]: Upload the dataset `SocraticChat` in [hugging face](https://huggingface.co/datasets/FreedomIntelligence/SocraticChat).
- [10/10/2023]: Update the [tech report v4](https://arxiv.org/abs/2308.11534v4).
- [10/08/2023]: The user simulator `UserGPT`, dataset `RealChat` and the respondent model `ReaLM` are renamed to `Socratic`, `SocraticChat`, and `PlatoLM` by Benyou Wang, the provider of 4 x A100s.
- [08/21/2023]: PlatoLM-7b Rank #1 on [AlpacaEval benchmark](https://tatsu-lab.github.io/alpaca_eval/) among 7B scale, achieving 81.94% win rates against text-davinci-003 (has entered into the official benchmark).
- [08/21/2023]: PlatoLM-7b Rank #1 on [MT-Bench benchmark](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) among 7B scale (hasn't entered into the official benchmark yet).
- [08/21/2023]: Release the [model weights](https://huggingface.co/FreedomIntelligence/PlatoLM-7b/tree/main).
- [08/21/2023]: Release the [tech report v1](https://arxiv.org/abs/2308.11534).

# ⚡ Introduction

Welcome to our realm🤗

We propose a new paradigm for training a user simulator.

After applying this paradigm to ShareGPT and LLaMA-7B, a novel user simulator, `Socratic`, emerged. Through iterative interactions between Socratic and gpt-3.5-turbo, a multi-round conversation dataset named `SocraticChat` was generated. Leveraging this dataset for fine-tuning LLAMA-7B-2 resulted in the `PlatoLM` model, which exhibits superior performance. 

With fewer samples(50.7K) distilled from gpt-3.5, shorter context length(2048), and smaller model scale(7B), we even beat GPT 3.5 in Alpaca-Eval benchmark.

<img src="https://github.com/FreedomIntelligence/PlatoLM/assets/73695787/b314a609-dfc6-4d6a-9795-3bf492f84c0c.png" width="700" height="250" alt="cool">

<img src="https://github.com/FreedomIntelligence/PlatoLM/assets/73695787/51141cbc-046a-4a55-b937-254e1155c06b.png" width="400" height="350" alt="cool">


# 📖 Methodology

The key to our idea is to `flip the chessboard`.

We just `mask the questions of real users` and accordingly, only `calculate their loss` for the purpose of `modifying the learning objective`.
In addition, we use `a dyadic prompt template` to instruct our backbone.

The main difference between us and other research is shown below.
![pipeline](https://github.com/FreedomIntelligence/PlatoLM/assets/73695787/ecd6156e-4125-4e3b-93a3-b9955cb740ce)

The pipeline can be analogous to `Socratic teaching`, which means teaching students via questioning. We argue that after learning the real human's high-quality instructions based on the knowledgeable llama backbone, more human-like LLMs will master the sophisticated teaching ability.
Therefore, we named the query model `Socratic`, which means the follower of Socrates.  Likewise, we labeled the dataset as `SocraticChat`, and the resulting model was dubbed `PlatoLM`.
<img src="https://github.com/FreedomIntelligence/PlatoLM/assets/73695787/5c60df0a-93a3-44bd-a6b3-fa4e2e73ad96.png" width="600" height="400" alt="analogy">

Experiments show that a more human-like questioning pattern in dynamic multi-round conversations can teach the response model better compared to static role-playing, which can be attributed to `the natural and rich topic structures of the questioning pattern from humans` in human-machine dialogue where they `hold topic dominance`. 

# 📄 Case Study

`The typical samples` for Socratic Dialogues and our dataset SocraticChat are shown below.
![sample2](https://github.com/FreedomIntelligence/PlatoLM/assets/73695787/22e3754d-a28c-4cf3-a7fb-517afa6ec41a)



# 🚀 Training

```shell
# To fine-tune Socratic
cd model/sft_socratic
bash scripts/sft_7b.sh 

# To fine-tune PlatoLM
cd model/sft_platolm
bash scripts/sft_7b.sh 
```

# 🧐 Inferencing

```shell
# To infer PlatoLM
python -m model.sft_platolm.source.deploy.cli --model FreedomIntelligence/PlatoLM-7b

# To infer Socratic
# The model's weights of Socratic has not been published yet. 
python -m model.sft_socratic.source.deploy.cli --model balabala
```

# 🎉 Acknowledgement

We are aware that our works are inspired by the following works, including but not limited to

- LLaMA: https://huggingface.co/meta-llama
- Self-instruct: https://github.com/yizhongw/self-instruct
- LLMZoo: https://github.com/FreedomIntelligence/LLMZoo

Without these, nothing could happen in this repository.

# 💭 Citation

```
@misc{kong2023platolm,
      title={PlatoLM: Teaching LLMs via a Socratic Questioning User Simulator}, 
      author={Chuyi Kong and Yaxin Fan and Xiang Wan and Feng Jiang and Benyou Wang},
      year={2023},
      eprint={2308.11534},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

We are from the School of Data Science, the Chinese University of Hong Kong, Shenzhen (CUHKSZ), and the Shenzhen Research Institute of Big Data (SRIBD).
