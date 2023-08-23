import abc

import torch
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
    )
except ImportError:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
    )
 
from utils import get_default_conv_template, SeparatorStyle

def get_gpu_memory(max_gpus=None):
    gpu_memory = []
    num_gpus = (
        torch.cuda.device_count()
        if max_gpus is None
        else min(max_gpus, torch.cuda.device_count())
    )

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024 ** 3)
            allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory

async def load_model(
        model_path, device, num_gpus, max_gpu_memory=None, debug=False,
):
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs["device_map"] = "auto"
                if max_gpu_memory is None:
                    kwargs[
                        "device_map"
                    ] = "sequential"  
                    available_gpu_memory = get_gpu_memory(num_gpus)
                    kwargs["max_memory"] = {
                        i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                        for i in range(num_gpus)
                    }
                else:
                    kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}
        print("init_kwargs", kwargs)
    else:
        raise ValueError(f"Invalid device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    if device == "cuda" and num_gpus == 1:
        model.to(device)
    
    if debug:
        print(model)

    return model, tokenizer

# control the termination
class GenerateStreamStopException(Exception):
    pass

@torch.inference_mode()
def generate_stream(model, tokenizer, params, device, context_len=2048, stream_interval=2):
    prompt = params["prompt"]
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    stop_token_ids = params.get("stop_ids", [tokenizer.eos_token_id])

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    l_prompt = len(tokenizer.decode(input_ids, skip_special_tokens=False))

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    for i in range(max_new_tokens):
        if i == 0:
            if model.config.is_encoder_decoder:
                encoder_outputs = model.encoder(
                    input_ids=torch.as_tensor([input_ids], device=device)
                )
                out = model(
                    torch.as_tensor([input_ids], device=device),
                    decoder_input_ids=torch.as_tensor(
                        [[model.generation_config.decoder_start_token_id]],
                        device=device,
                    ),
                    encoder_outputs=encoder_outputs,
                    use_cache=True,
                )
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
        else:
            if model.config.is_encoder_decoder:
                out = model(
                    input_ids=torch.as_tensor([input_ids], device=device),
                    use_cache=True,
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=torch.as_tensor([[token]], device=device),
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]

        if device == "mps":
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=False)
            if stop_str:
                pos = output.rfind(stop_str, l_prompt)
                if pos != -1:
                    output = output[l_prompt:pos]
                    stopped = True
                else:
                    output = output[l_prompt:]
                yield output 
            else:
                raise NotImplementedError

        if stopped:
            break

        # control the termination
        if len(output_ids) > context_len:
            raise GenerateStreamStopException()

    del past_key_values

class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream):
        """Stream output."""
        
class SimpleChatIO(ChatIO):
    def stream_output(self, output_stream):
        pre = 0
        try:
            for outputs in output_stream:
                outputs = outputs.strip()
                outputs = outputs.split(" ")
                now = len(outputs) - 1
                if now > pre:
                    pre = now
        except GenerateStreamStopException:
            return None
        return " ".join(outputs)

def produce_history( 
        model, tokenizer, history, 
        model_path: str,
        device: str="cuda",
        temperature: float=0.7,
        max_new_tokens: int=512,
        chatio: ChatIO=SimpleChatIO(),
        ):
        
        conv = get_default_conv_template().copy()
        if history != None:
            conv.messages = history 

        conv.append_message(conv.roles[0], None)
       
        generate_stream_func = generate_stream
        prompt = conv.get_prompt()
        params = {
            "model": model_path,
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else None,
        }
        output_stream = generate_stream_func(model, tokenizer, params, device)
        q = chatio.stream_output(output_stream)
        # control the termination
        if not q:
            return None
        conv.messages[-1][-1] = q.strip()
    
        return conv.messages




