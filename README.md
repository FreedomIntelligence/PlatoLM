# PlatoLM: Teaching LLMs via a Socratic Questioning User Simulator

# 	:calendar: Future Plan
- [10/08/2023]
- will update the tech report v1.1 in 3 days.
- will update the 13b version in a week.
- will reconstruct the code in a month.
- if you are interested in our work, welcome to provide us with gpt APIs and GPUs!

# ✨ Latest News
- [08/21/2023]: Rank #1 on [AlpacaEval benchmark](https://tatsu-lab.github.io/alpaca_eval/) among 7B scale, achieving 81.94% win rates against text-davinci-003 (has entered into the official benchmark).
- [08/21/2023]: Rank #1 on [MT-Bench benchmark](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) among 7B scale (hasn't entered into the official benchmark yet).
- [08/21/2023]: Release the [model weights](https://huggingface.co/FreedomIntelligence/PlatoLM-7b/tree/main).
- [08/21/2023]: Release the [tech report v1.0](https://arxiv.org/abs/2308.11534).

# ⚡ Introduction
Welcome to our realm:hugs:

We propose a new paradigm for training a user simulator. 

After applying this paradigm to ShareGPT and LLaMA-7B, a novel user simulator, `Socratic`, emerged. Through iterative interactions between Socratic and gpt-3.5-turbo, a multi-round conversation dataset named `SocraticChat` was generated. Leveraging this dataset for fine-tuning LLAMA-7B-2 resulted in the `PlatoLM` model, which exhibits superior performance. 

<img src="https://github.com/FreedomIntelligence/PlatoLM/assets/73695787/253152f0-3262-4db8-9d4f-c66aab9b4323.png" width="500" height="200" alt="cool">

# :book: Methodology
The key to our idea is to `flip the chessboard`.

We just `mask the questions of real users` and accordingly, only `calculate their loss` for the purpose of `modifying the learning objective`.
In addition, we use `a dyadic prompt template` to instruct our backbone.

The main difference between us and other research is shown below.
![pipeline](https://github.com/FreedomIntelligence/PlatoLM/assets/73695787/ecd6156e-4125-4e3b-93a3-b9955cb740ce)

The pipeline can be analogous to `Socratic teaching`, which means `teaching students via questioning from shallow to deeper`. We argue that after `learning the real human's high-quality instructions` based on `the knowledgeable llama backbone`, Socratic shows the `sophisticated pedagogical ability to its students`, PlatoLM.
Hence, the query model we named as `Socratic`, which means the follower of Socrates. The dataset was debudded with `ScoraticChat` and the resultant model was named `PlatoLM`.
<img src="https://github.com/FreedomIntelligence/PlatoLM/assets/73695787/5e735e31-3561-4e7f-9770-5e6f205dfbd1.png" width="500" height="330" alt="analogy">



# :page_facing_up: Case Study
Experiments show that Socratic learned some `natural patterns of mindset for human-computer interaction` and it can `ask questions progressively`, as opposed to WizardLM which needs to undergo several rounds of evolution and filtering. `The typical samples` for Socratic Dialogues and our dataset SocraticChat are shown below.
<img src="https://github.com/FreedomIntelligence/PlatoLM/assets/73695787/e4da7bdc-2102-4df7-9f31-eb3e8ca46c24.png" style="max-width: 500px; max-height: 1000px;"  alt="sample">

 
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
will update in 3 days!
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
