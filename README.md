# Large Language Model as a User Simulator

# ✨ Latest News
- [08/21/2023]: Release the [model weights](https://huggingface.co/FreedomIntelligence/Realm-7b/tree/main).
- [08/21/2023]: Release the [tech report](https://arxiv.org/abs/2308.11534v1).

# ⚡ Introduction
Welcome to our realm:hugs:

We propose a new paradigm for training a user simulator. 

After applying this paradigm to ShareGPT and LLaMA-7B, a novel user simulator, `UserGPT`, emerged. Through iterative interactions between UserGPT and gpt-3.5-turbo, a multi-round conversation dataset named `RealChat` was generated. Leveraging this dataset for fine-tuning LLAMA-7B-2 resulted in the `ReaLM` model, which exhibits superior performance compared to LLaMA-2-7B-chat and Vicuna in MT-Bench. Impressively, this improvement was achieved using a modest 50.7K samples.

<img src="https://github.com/FreedomIntelligence/ReaLM/assets/73695787/808bcc05-dcae-4fa4-a11e-2c5496ae79b3" alt="performance" width="50%" height="50%">

The dataset was dubbed `RealChat` due to its human-like user side. UserGPT's versatility in switching between raising novel questions and incorporating single-round dialogs as conversational instructions to create domain-specific datasets led to the naming of the answering model as `ReaLM`.

# :book: Methodology
The key to our idea is to `flip the chessboard` (:chess_pawn:from an AVG —— うみねこのなく頃に).

<img src="https://github.com/FreedomIntelligence/ReaLM/assets/73695787/e034f4db-5248-437e-83dd-aa3a940add70" alt="key" width="50%" height="50%">


We just `mask the questions of real users` and accordingly, only `calculate their loss` for the purpose of `modifying the learning objective`.
In addition, we use `a dyadic prompt template` to instruct our backbone.

The main difference between us and other research is shown below.
![haha](https://github.com/FreedomIntelligence/ReaLM/assets/73695787/31baa406-e8c0-4fe4-854c-41f798ed8d52)

# 🚀 Training
```shell
# To fine-tune UserGPT, you can use the following command
bash scripts/SFT_UserGPT_7B.sh
# To fine-tune ReaLM, you can use the following command
bash scripts/SFT_ReaLM_7B.sh
```
# 🧐 Inferencing
```shell
# To infer ReaLM, you can use the following command
python -m model.sft_realm.source.deploy.cli --model FreedomIntelligence/ReaLM-7b
# To infer UserGPT, you can use the following command
# The model's weights of UserGPT has not been published yet. 
python -m model.sft_usergpt.source.deploy.cli --model balabala
```

# :tada: Acknowledgement

We are aware that our works are inspired by the following works, including but not limited to

- llama: https://huggingface.co/meta-llama
- Self-instruct: https://github.com/yizhongw/self-instruct
  
Without these, nothing could happen in this repository.

# 💭 Citation
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
