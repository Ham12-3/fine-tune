"""Prompt builders shared by every model, so Claude and the small model see identical text.

The system prompt carries everything that is the same for every request (instructions, the 77 labels with
descriptions, and the few-shot block in few-shot mode). It is byte-stable, so Claude can cache it. The user turn
is only the customer's message.
"""

from __future__ import annotations

INSTRUCTIONS = (
    "You classify messages that customers send to a digital bank's support chat.\n"
    "Choose the single intent from the list below that best matches the customer's message. "
    "Answer with that one intent name, exactly as written in the list."
)

# Short prompt for the fine-tuned model (Phase 5): the label list is learnt, not given.
FINETUNE_INSTRUCTION = "Classify this banking customer message into its intent."


def label_block(labels: list[dict]) -> str:
    return "\n".join(f"- {lab['name']}: {lab['description']}" for lab in labels)


def few_shot_block(examples: list[dict]) -> str:
    return "\n\n".join(f"Message: {ex['text'].strip()}\nIntent: {ex['label']}" for ex in examples)


def system_prompt(labels: list[dict], few_shot: list[dict] | None = None) -> str:
    parts = [INSTRUCTIONS, "Intents (name: description):\n" + label_block(labels)]
    if few_shot:
        parts.append("Labelled examples:\n\n" + few_shot_block(few_shot))
    return "\n\n".join(parts)


def user_message(text: str) -> str:
    return text.strip()


def finetune_prompt(text: str) -> str:
    return f"{FINETUNE_INSTRUCTION}\nMessage: {text.strip()}"


def label_schema(labels: list[dict]) -> dict:
    """JSON schema for Claude's structured output: {"label": one of the 77 names}."""
    return {
        "type": "object",
        "properties": {"label": {"type": "string", "enum": [lab["name"] for lab in labels]}},
        "required": ["label"],
        "additionalProperties": False,
    }
