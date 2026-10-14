from svf.models import gguf_infer


def test_chat_prompt_matches_training_format():
    p = gguf_infer.chat_prompt("  I am still waiting on my card?\n")
    assert p == ("<|im_start|>user\nClassify this banking customer message into its intent.\n"
                 "Message: I am still waiting on my card?<|im_end|>\n"
                 "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert gguf_infer.STOP == ["<|im_end|>", "<|endoftext|>"]
