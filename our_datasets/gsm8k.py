"""
GSM8K dataset for open-ended math generation evaluation.
Not a BaseTask subclass — uses generation-style evaluation (greedy decode + exact match).
"""
import random
import re

from datasets import load_dataset


class GSM8KDataset:
    task_type = "generation"
    task_name = "gsm8k"

    def __init__(self, split='train', max_data_num=None, seed=42):
        random.seed(seed)

        if split == 'test':
            hf_split = 'test'
            data = list(load_dataset('openai/gsm8k', 'main', split=hf_split, keep_in_memory=True))
        else:
            # train and validation both come from the HF train split
            raw = list(load_dataset('openai/gsm8k', 'main', split='train', keep_in_memory=True))
            val_size = max(1, len(raw) // 10)  # last 10% → validation
            if split == 'validation':
                data = raw[-val_size:]
            else:
                data = raw[:-val_size]

        if max_data_num is not None and max_data_num < len(data):
            data = random.sample(data, max_data_num)

        self.all_data = data
        print(f"GSM8K ({split}): {len(self.all_data)} samples")

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_query(item):
        return f"Question: {item['question'].strip()}\nAnswer:"

    @staticmethod
    def get_answer(item):
        """Extract the final numerical answer after ####."""
        match = re.search(r'####\s*([\d,.\-]+)', item['answer'])
        if match:
            return match.group(1).replace(',', '')
        # Fallback: last number in the answer
        numbers = re.findall(r'[\d,.\-]+', item['answer'])
        return numbers[-1].replace(',', '') if numbers else item['answer'].strip()

    def build_demo(self, items, sep='\n\n'):
        """Build CoT few-shot demonstration string from a list of items."""
        parts = []
        for item in items:
            q = f"Question: {item['question'].strip()}"
            answer_text = item['answer'].strip()
            match = re.search(r'####\s*([\d,.\-]+)', answer_text)
            if match:
                final_num = match.group(1).replace(',', '')
                reasoning = answer_text[:match.start()].strip()
                a = f"Let's think step by step.\n{reasoning}\nTherefore, the answer is {final_num}."
            else:
                final_num = self.get_answer(item)
                a = f"Let's think step by step.\nTherefore, the answer is {final_num}."
            parts.append(f"{q}\nAnswer: {a}")
        return sep.join(parts) + sep

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def parse_prediction(text):
        # Truncate if model continues generating next Q-A pair
        text = text.split('\n\nQuestion:')[0].strip()
        # Priority: "Therefore, the answer is X"
        match = re.search(r'[Tt]herefore,?\s+the\s+answer\s+is\s+(-?[\d,]+\.?\d*)', text)
        if match:
            return match.group(1).replace(',', '')
        # Fallback: last "= X"
        eq_nums = re.findall(r'=\s*(-?[\d,]+\.?\d*)', text)
        if eq_nums:
            return eq_nums[-1].replace(',', '')
        first_line = text.split('\n')[0].strip()
        nums = re.findall(r'-?[\d,]+\.?\d*', first_line)
        if nums:
            return nums[-1].replace(',', '')
        nums = re.findall(r'-?[\d,]+\.?\d*', text)
        return nums[-1].replace(',', '') if nums else text.strip()

    @staticmethod
    def compare(pred, gold):
        """Numerical equality with string fallback."""
        pred = str(pred).strip().replace(',', '')
        gold = str(gold).strip().replace(',', '')
        try:
            return abs(float(pred) - float(gold)) < 1e-6
        except ValueError:
            return pred.lower() == gold.lower()

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.all_data[idx]
