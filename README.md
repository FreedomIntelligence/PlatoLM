# ReaLM
Welcome to our realm.

We propose a new paradigm for training a user simulator. 

After applying this paradigm to ShareGPT and LLaMA-7B, a novel user simulator, `UserGPT`, emerged. Through iterative interactions between UserGPT and gpt-3.5-turbo, a multi-round conversation dataset named `RealChat` was generated. Leveraging this dataset for fine-tuning LLAMA-7B-2 resulted in the `ReaLM` model, which exhibits superior performance compared to LLaMA-2-7B-chat and Vicuna in MT-Bench. Impressively, this improvement was achieved using a modest 50.7K samples.

<img src="https://github.com/FreedomIntelligence/ReaLM/assets/73695787/808bcc05-dcae-4fa4-a11e-2c5496ae79b3" alt="performance" width="50%" height="50%">

The dataset was dubbed `RealChat` due to its human-like user side. UserGPT's versatility in switching between raising novel questions and incorporating single-round dialogs as conversational instructions to create domain-specific datasets led to the naming of the answering model as `ReaLM`.

## Methodology
The key to our idea is to flip the chessboard.

<img src="https://github.com/FreedomIntelligence/ReaLM/assets/73695787/e034f4db-5248-437e-83dd-aa3a940add70" alt="key" width="50%" height="50%">


We just mask the questions of real users and accordingly, only calculate their loss for the purpose of modifying the learning objective.
In addition, we use a dyadic prompt template to instruct our backbone.

The main difference between us and other research is shown below.
![haha](https://github.com/FreedomIntelligence/ReaLM/assets/73695787/31baa406-e8c0-4fe4-854c-41f798ed8d52)

## Links
The reference paper is available at the following links:

https://arxiv.org/abs/2308.11534v1


The ReaLM model is available at the following links:

https://huggingface.co/FreedomIntelligence/Realm-7b/tree/main


