import argparse

from inference import chat_loop, ChatIO


class SimpleChatIO(ChatIO):
    def prompt_for_input(self, role) -> str:
        return input(f"{role}: ")

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            outputs = outputs.strip()
            outputs = outputs.split(" ")
            now = len(outputs) - 1
            if now > pre:
                print(" ".join(outputs[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(outputs[pre:]), flush=True)
        return " ".join(outputs)

def main(args):
    chatio = SimpleChatIO()
    try:
        chat_loop(
            args.model_path,
            args.device,
            args.num_gpus,
            args.max_gpu_memory,
            args.conv_template,
            args.temperature,
            args.max_new_tokens,
            chatio,
            args.debug,
        )
    except KeyboardInterrupt:
        print("exit...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="FreedomIntelligence/ReaLM",
        help="The path to the weights",
    )
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="The maximum memory per gpu. Use a string like '13Gib'",
    )
    parser.add_argument("--conv-template", type=str, default=None, help="Conversation prompt template.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
