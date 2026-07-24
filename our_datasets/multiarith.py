"""
MultiArith dataset for open-ended arithmetic generation evaluation.
Not a BaseTask subclass — uses generation-style evaluation (greedy decode + exact match).
Only a 'train' split exists on HF, so we split it manually.
"""
import random
import re

from datasets import load_dataset


class MultiArithDataset:
    task_type = "generation"
    task_name = "multiarith"

    def __init__(self, split='train', max_data_num=None, seed=42, is_instruct=True):
        random.seed(seed)
        self.is_instruct = is_instruct

        raw = list(load_dataset('ChilleD/MultiArith', split='train', keep_in_memory=True))

        # Deterministic shuffle before split so train/val/test don't overlap
        rng = random.Random(seed)
        shuffled = raw[:]
        rng.shuffle(shuffled)

        n = len(shuffled)
        val_size = max(1, n // 10)   # 10% validation
        test_size = max(1, n // 10)  # 10% test

        if split == 'test':
            data = shuffled[-test_size:]
        elif split == 'validation':
            data = shuffled[-(val_size + test_size):-test_size]
        else:
            data = shuffled[:n - val_size - test_size]

        if max_data_num is not None and max_data_num < len(data):
            data = random.sample(data, max_data_num)

        self.all_data = data
        print(f"MultiArith ({split}): {len(self.all_data)} samples")

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def get_query(self, item):
        suffix = "Let's think step by step." if self.is_instruct else ""
        suffix = f" {suffix}" if suffix else ""
        return f"Question: {item['question'].strip()}\nAnswer:{suffix}"

    @staticmethod
    def get_answer(item):
        return str(item['final_ans']).strip().replace(',', '')

    def build_demo(self, items, sep='\n\n'):
        """Build few-shot demonstration string (answer-only; no CoT data in dataset)."""
        parts = []
        for item in items:
            q = f"Question: {item['question'].strip()}"
            a = self.get_answer(item)
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
