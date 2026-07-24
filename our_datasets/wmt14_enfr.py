"""
WMT14 English->French translation dataset for open-ended text generation evaluation.
Not a BaseTask subclass — generation-style (like GSM8K/MultiArith).

Note: HF `wmt16`'s `fr-en` config does not exist (only cs/de/fi/ro/ru-en).
`wmt14` does have `fr-en` and is the config GPT-3 Table H.1 actually reports
numbers for, so we use that.
"""
import random
import re

from datasets import load_dataset

# The only two strings that can start the *next* turn if the model is
# following our template correctly: get_query() always formats the next
# item as "English: ...", and build_demo()'s separator is "\n\n". This is
# not a guess at what a "language label" looks like — it's the literal
# contract of the template we wrote. If the model instead drifts into some
# other unanticipated continuation (e.g. a different language), that text
# is left in the prediction and scored as-is: it's a genuine formatting
# failure, not something to launder out post-hoc.
_NEXT_TURN_PATTERN = r'\nEnglish:|\n\n'


class WMT14EnFrDataset:
    task_type = "generation"
    task_name = "wmt14_enfr"
    max_new_tokens = 64

    def __init__(self, split='train', max_data_num=None, seed=42):
        random.seed(seed)

        # IMPORTANT: must use streaming=True. wmt14/fr-en's non-streaming
        # loader downloads the *entire* shared raw archive bundle (Europarl,
        # Common Crawl, UN corpus, Giga-FREN, ...) before it can hand back
        # even the small validation/test splits — tens of GB. streaming=True
        # lazily fetches only the file(s) backing the requested split.
        #
        # demo/train-anchor/validation all come from the small 'validation'
        # split (3000 pairs, newstest2013); the held-out 'test' split (3003
        # pairs, newstest2014) is reserved for final evaluation. The true
        # ~40M-pair 'train' config is never touched.
        if split == 'test':
            data = list(load_dataset('wmt14', 'fr-en', split='test', streaming=True))
        else:
            raw = list(load_dataset('wmt14', 'fr-en', split='validation', streaming=True))
            random.Random(seed).shuffle(raw)
            val_size = max(1, len(raw) // 10)  # last 10% -> validation
            if split == 'validation':
                data = raw[-val_size:]
            else:
                data = raw[:-val_size]

        if max_data_num is not None and max_data_num < len(data):
            data = random.sample(data, max_data_num)

        self.all_data = data
        print(f"WMT14 En-Fr ({split}): {len(self.all_data)} samples")

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_query(item):
        return f"English: {item['translation']['en'].strip()}\nFrench:"

    @staticmethod
    def get_answer(item):
        return item['translation']['fr'].strip()

    def build_demo(self, items, sep='\n\n'):
        parts = []
        for item in items:
            q = f"English: {item['translation']['en'].strip()}"
            a = item['translation']['fr'].strip()
            parts.append(f"{q}\nFrench: {a}")
        return sep.join(parts) + sep

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def parse_prediction(text):
        # Truncate at the first occurrence of either literal template
        # boundary (re.search already returns the leftmost match among the
        # '|' alternatives, i.e. whichever comes first).
        m = re.search(_NEXT_TURN_PATTERN, text)
        return (text[:m.start()] if m else text).strip()

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.all_data[idx]
