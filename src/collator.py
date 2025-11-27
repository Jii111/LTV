from dataclasses import dataclass
from typing import Union, Optional, Any

import torch
from transformers import PreTrainedTokenizerBase, BatchEncoding
from transformers.utils import PaddingStrategy

from I2CL.my_datasets import BaseTask

__all__ = [
    "DataCollatorForSPT"
]


@dataclass
class DataCollatorForSPT:
    """
    Data collator that will format ICL.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.0 (Volta).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", or "pt".
    """

    tokenizer: PreTrainedTokenizerBase
    dataset_class: BaseTask
    shot_num: Optional[int] = None
    seed: Optional[int] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    candidate_indices: Optional[torch.LongTensor] = None

    def __post_init__(self):
        """Initialize candidate indices once during initialization."""
        self.candidate_indices = self._get_candidate_indices()

    def _get_candidate_indices(self) -> torch.LongTensor:
        """
        Get candidate token indices for labels from the dataset.
        Returns a tensor of token IDs corresponding to each label option.
        """
        ans_txt_list = self.dataset_class.get_dmonstration_template()['options']
        candidate_indices = []

        for ans_txt in ans_txt_list:
            # Add space prefix for Qwen models
            if 'qwen' in self.tokenizer.__class__.__name__.lower():
                ans_txt = ' ' + ans_txt

            ans_tok = self.tokenizer.encode(ans_txt, add_special_tokens=False)[0]  # use the first token
            candidate_indices.append(ans_tok)

        return torch.LongTensor(candidate_indices)

    def __call__(self, features: list[dict[str, Any]]) -> BatchEncoding:
        zero_shot_inputs = [input_str + " " + ans_list[label]
                            for f in features
                            for input_str, ans_list, label in [self.dataset_class.apply_template(f)]]

        few_shot_inputs = [self.dataset_class.gen_few_shot_demonstration(
            tokenizer=self.tokenizer,
            shot_num=self.shot_num,
            seed=self.seed,
        )[0] + "\n" + z for z in zero_shot_inputs]

        few_shot_tokenized_inputs = self.tokenizer(
            text=few_shot_inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors
        )

        zero_shot_tokenized_inputs = self.tokenizer(
            text=zero_shot_inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors
        )

        # Remove token before EOS (keep EOS)
        few_shot_input_ids = few_shot_tokenized_inputs["input_ids"][:, :-1]
        few_shot_attention_mask = few_shot_tokenized_inputs["attention_mask"][:, :-1]

        zero_shot_input_ids = zero_shot_tokenized_inputs["input_ids"][:, :-1]
        zero_shot_attention_mask = zero_shot_tokenized_inputs["attention_mask"][:, :-1]

        # Collect answer tokens (the last token that was removed)
        answer_token = zero_shot_tokenized_inputs["input_ids"][:, -1]

        return BatchEncoding(
            {
                "few_shot_input_ids": few_shot_input_ids,
                "few_shot_attention_mask": few_shot_attention_mask,
                "zero_shot_input_ids": zero_shot_input_ids,
                "zero_shot_attention_mask": zero_shot_attention_mask,
                "candidate_indices": self.candidate_indices,
                "answer_token": answer_token,
            }
        )


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from I2CL.my_datasets.agnews import AGNews

    print("=" * 50)
    print("Testing DataCollatorForSPT with AGNews")
    print("=" * 50)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
    tokenizer.padding_side = "left"
    # Load dataset
    agnews = AGNews(split='train', task_name="agnews")

    # Create collator
    collator = DataCollatorForSPT(
        tokenizer=tokenizer,
        dataset_class=agnews,
        shot_num=30,
        seed=42
    )

    # Get sample features
    features = [agnews.all_data[i] for i in range(3)]

    print("\n[Input Features]")
    for i, f in enumerate(features):
        print(f"{i}: {f['text'][:50]}... (label={f['label']})")

    # Call collator
    batch = collator(features)

    print("\n[Output Batch Keys]")
    print(batch.keys())

    print("\n[Few-shot Input IDs Shape]")
    print(batch["few_shot_input_ids"].shape)

    print("\n[Zero-shot Input IDs Shape]")
    print(batch["zero_shot_input_ids"].shape)

    print("\n[First Few-shot Example (decoded)]")
    print(tokenizer.decode(batch["few_shot_input_ids"][0]))

    print("\n[First Zero-shot Example (decoded)]")
    print(tokenizer.decode(batch["zero_shot_input_ids"][0]))
