from typing import List
from enum import auto, Enum
from dataclasses import dataclass

import transformers

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

class SeparatorStyle(Enum):
    SINGLE = auto()
    TWO = auto()

@dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "</s>"

    skip_next: bool = False

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep
        }

# you can try it to test how can terminate the conversations by human
sft_usergpt_end_test = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
            "The human can ask further questions based on assistant's answers above, or he can directly ask questions without context."
            "When the human thinks that the assistant's answer is clear enough for him, the human will end the conversation by replying <e>, where <e> is a word that has no semantic meaning and simply means that the human considers that the conversation should end at this point.\n\n",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="</s>",
)

sft_usergpt_final = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
            "The human can ask further questions based on previous conversations, or he can directly ask brand new questions without any conversations as context.\n\n",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="</s>",
)

default_conversation = sft_usergpt_final

def get_default_conv_template():
    return default_conversation

if __name__ == "__main__":
    print(default_conversation.get_prompt())
