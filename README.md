# PlatoLM: Teaching LLMs via a Socratic Questioning User Simulator

# ✨ Latest News
- [09/28/2023]: Rank second on [AlpacaEval benchmark](https://tatsu-lab.github.io/alpaca_eval/) among 7b scale and beat some 13b models, achieving 81.94% win rates against text-davinci-003.
- [08/21/2023]: Rank top on [MT-Bench](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) among 7B scale.
- [08/21/2023]: Release the [model weights](https://huggingface.co/FreedomIntelligence/PlatoLM-7b/tree/main).
- [08/21/2023]: Release the [tech report v1.0](https://arxiv.org/abs/2308.11534).

# ⚡ Introduction
Welcome to our realm:hugs:

We propose a new paradigm for training a user simulator. 

After applying this paradigm to ShareGPT and LLaMA-7B, a novel user simulator, `Socratic`, emerged. Through iterative interactions between Socratic and gpt-3.5-turbo, a multi-round conversation dataset named `SocraticChat` was generated. Leveraging this dataset for fine-tuning LLAMA-7B-2 resulted in the `PlatoLM` model, which exhibits superior performance compared to LLaMA-2-7B-chat and Vicuna in MT-Bench and Alpaca-Eval. Impressively, this improvement was achieved using only 50.7K samples.

~~The dataset was dubbed `RealChat` due to its human-like user side. Socratic's versatility in switching between raising novel questions and incorporating single-round dialogs as conversational instructions to create domain-specific datasets led to the naming of the answering model as `ReaLM`.~~

# :book: Methodology
The key to our idea is to `flip the chessboard`.

We just `mask the questions of real users` and accordingly, only `calculate their loss` for the purpose of `modifying the learning objective`.
In addition, we use `a dyadic prompt template` to instruct our backbone.

The main difference between us and other research is shown below.
![pipeline](https://github.com/FreedomIntelligence/PlatoLM/assets/73695787/ecd6156e-4125-4e3b-93a3-b9955cb740ce)


# 🚀 Training
```shell
# To fine-tune Socratic, you can use the following command
bash model/sft_socratic/scripts/sft_socratic_7b.sh
# To fine-tune PlatoLM, you can use the following command
bash model/sft_platolm/scripts/sft_platolm_7b.sh
```
# 🧐 Inferencing
```shell
# To infer PlatoLM, you can use the following command
python -m model.sft_platolm.source.deploy.cli --model FreedomIntelligence/PlatoLM-7b
# To infer Socratic, you can use the following command
# The model's weights of Socratic has not been published yet. 
python -m model.sft_socratic.source.deploy.cli --model balabala
```

# :tada: Acknowledgement

We are aware that our works are inspired by the following works, including but not limited to

- llama: https://huggingface.co/meta-llama
- Self-instruct: https://github.com/yizhongw/self-instruct
  
Without these, nothing could happen in this repository.

# 💭 Citation
will update in a week!
```
@misc{kong2023large,
      title={Large Language Model as a User Simulator}, 
      author={Chuyi Kong and Yaxin Fan and Xiang Wan and Feng Jiang and Benyou Wang},
      year={2023},
      eprint={2308.11534},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
We are from the School of Data Science, the Chinese University of Hong Kong, Shenzhen (CUHKSZ), and the Shenzhen Research Institute of Big Data (SRIBD).
