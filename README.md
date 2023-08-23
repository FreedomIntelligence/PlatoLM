# ReaLM
Welcome to our Realm.

We propose a new paradigm for training a user simulator. 

When using this paradigm on ShareGPT and LLaMA-7B, we got a brand new user simulator `UserGPT`.  When calling the UserGPT and gpt-3.5-turbo iteratively, we generated a multi-round conversation dataset `RealChat`. When fine-tuning LLAMA-7B-2 with it, the performance of the resulting model `ReaLM` surpass LLaMA-2-7B-chat and Vicuna with only 50.7K samples in MT-Bench.

![image](https://github.com/FreedomIntelligence/ReaLM/assets/73695787/808bcc05-dcae-4fa4-a11e-2c5496ae79b3){:width="50%" height="50%"}

The reference paper is available at the following links:
https://arxiv.org/abs/2308.11534v1

## Methodology
The key idea of our methodology is to flip the chessboard.

We just mask the questions of real users and accordingly, only calculate their loss for the purpose of modifying the learning objective.
In addition, we use a dyadic prompt template to instruct our backbone.

The main difference between us and other research is shown below.
![haha](https://github.com/FreedomIntelligence/ReaLM/assets/73695787/31baa406-e8c0-4fe4-854c-41f798ed8d52)

