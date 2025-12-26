"""
Tokenizer Analysis for Few-shot ICL

Compares:
1. Label tokens as they actually appear in demonstration
2. Label tokens from evaluator logic (current code)
"""

import argparse
from transformers import AutoTokenizer
import my_datasets as md


def get_all_datasets():
    return [k for k in md.target_datasets.keys() if k != 'emo']


def find_answer_in_demo(demo_tokens, answer_tokens):
    """Find where answer_tokens appear in demo_tokens, return the tokens at that position."""
    for i in range(len(demo_tokens) - len(answer_tokens) + 1):
        if demo_tokens[i:i+len(answer_tokens)] == answer_tokens:
            return i, demo_tokens[i:i+len(answer_tokens)]
    return None, None


def analyze_dataset(tokenizer, dataset_name):
    """Analyze a single dataset."""
    dataset = md.get_dataset(dataset_name, split='train', max_data_num=100, seed=42)
    template = dataset.get_dmonstration_template()
    options = template['options']

    # Get the prefix before label (e.g., "Sentiment:", "Type:", "Label:")
    input_template = template['input']
    # Extract the last part after newline (e.g., "Sentiment:" from "Review: {text}\nSentiment:")
    label_prefix = input_template.split('\n')[-1] if '\n' in input_template else input_template

    # Generate actual demonstration
    shot_num = len(options)
    demon, demon_list, indices = dataset.gen_few_shot_demonstration(
        tokenizer=tokenizer,
        shot_num=shot_num,
        max_demonstration_tok_len=10000,
        add_extra_query=False,
        example_separator='\n',
        return_data_index=True,
        seed=42
    )

    # Tokenize demonstration
    demo_tokens = tokenizer.encode(demon, add_special_tokens=False)

    tokenizer_name = tokenizer.__class__.__name__.lower()
    results = []

    for label, ans_txt in enumerate(options):
        # === Evaluator logic (current code) ===
        if 'gpt' in tokenizer_name or 'qwen' in tokenizer_name:
            eval_input = ' ' + ans_txt
        else:
            eval_input = ans_txt

        eval_toks = tokenizer.encode(eval_input, add_special_tokens=False)
        eval_tok1 = eval_toks[0]
        eval_tok2 = eval_toks[1] if len(eval_toks) > 1 else None
        eval_dec1 = tokenizer.decode([eval_tok1])
        eval_dec2 = tokenizer.decode([eval_tok2]) if eval_tok2 else None

        # === Actual demo: find ' answer\n' pattern ===
        # In demo, answer appears as: "Sentiment: positive\n"
        # So we search for ' positive\n' or ' positive'
        search_patterns = [
            ' ' + ans_txt + '\n',
            ' ' + ans_txt,
        ]

        demo_tok1 = None
        demo_tok2 = None
        demo_dec1 = None
        demo_dec2 = None
        found_pattern = None

        for pattern in search_patterns:
            pattern_toks = tokenizer.encode(pattern, add_special_tokens=False)
            pos, found_toks = find_answer_in_demo(demo_tokens, pattern_toks)
            if pos is not None:
                demo_tok1 = found_toks[0]
                demo_tok2 = found_toks[1] if len(found_toks) > 1 else None
                demo_dec1 = tokenizer.decode([demo_tok1])
                demo_dec2 = tokenizer.decode([demo_tok2]) if demo_tok2 else None
                found_pattern = pattern
                break

        # If not found with exact match, try tokenizing ' answer' and check first token
        if demo_tok1 is None:
            space_ans_toks = tokenizer.encode(' ' + ans_txt, add_special_tokens=False)
            demo_tok1 = space_ans_toks[0]
            demo_tok2 = space_ans_toks[1] if len(space_ans_toks) > 1 else None
            demo_dec1 = tokenizer.decode([demo_tok1])
            demo_dec2 = tokenizer.decode([demo_tok2]) if demo_tok2 else None
            found_pattern = "NOT FOUND (fallback)"

        match = eval_tok1 == demo_tok1

        results.append({
            'option': ans_txt,
            'eval_tok1': eval_tok1,
            'eval_tok2': eval_tok2,
            'eval_dec1': eval_dec1,
            'eval_dec2': eval_dec2,
            'demo_tok1': demo_tok1,
            'demo_tok2': demo_tok2,
            'demo_dec1': demo_dec1,
            'demo_dec2': demo_dec2,
            'match': match,
            'found_pattern': found_pattern,
            'label_prefix': label_prefix
        })

    return results, label_prefix


def analyze_all_datasets(tokenizer):
    """Analyze all datasets."""
    all_results = {}
    for dataset_name in get_all_datasets():
        try:
            results, label_prefix = analyze_dataset(tokenizer, dataset_name)
            all_results[dataset_name] = {'results': results, 'label_prefix': label_prefix}
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            continue
    return all_results


def print_table(tokenizer, all_results):
    """Print results as a formatted table."""
    tokenizer_name = tokenizer.__class__.__name__
    is_space_added = 'gpt' in tokenizer_name.lower() or 'qwen' in tokenizer_name.lower()

    print(f"\n{'='*180}")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Evaluator adds space: {'YES' if is_space_added else 'NO'}")
    print(f"{'='*180}\n")

    # Escape function
    def escape(s):
        if s is None:
            return None
        return s.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')

    # Header
    print(f"| {'Dataset':<13} | {'Label Prefix':<15} | {'Option':<14} | {'Eval tok1':<20} | {'Eval tok2':<20} | {'Demo tok1':<20} | {'Demo tok2':<20} | {'Match':<5} |")
    print(f"|{'-'*15}|{'-'*17}|{'-'*16}|{'-'*22}|{'-'*22}|{'-'*22}|{'-'*22}|{'-'*7}|")

    for dataset_name, data in all_results.items():
        results = data['results']
        label_prefix = data['label_prefix']
        first_row = True
        for r in results:
            ds_col = dataset_name if first_row else ""
            prefix_col = label_prefix if first_row else ""

            eval_tok1_col = f"{r['eval_tok1']}(`{escape(r['eval_dec1'])}`)"
            eval_tok2_col = f"{r['eval_tok2']}(`{escape(r['eval_dec2'])}`)" if r['eval_tok2'] else "-"
            demo_tok1_col = f"{r['demo_tok1']}(`{escape(r['demo_dec1'])}`)"
            demo_tok2_col = f"{r['demo_tok2']}(`{escape(r['demo_dec2'])}`)" if r['demo_tok2'] else "-"
            match_col = "✓" if r['match'] else "✗"

            print(f"| {ds_col:<13} | {prefix_col:<15} | {r['option']:<14} | {eval_tok1_col:<20} | {eval_tok2_col:<20} | {demo_tok1_col:<20} | {demo_tok2_col:<20} | {match_col:<5} |")
            first_row = False

    # Summary
    total = sum(len(data['results']) for data in all_results.values())
    matches = sum(sum(1 for r in data['results'] if r['match']) for data in all_results.values())
    mismatches = total - matches

    print(f"\n{'='*180}")
    print(f"SUMMARY: {matches}/{total} match, {mismatches} mismatches")
    if mismatches > 0:
        print(f"⚠️  WARNING: Evaluator token IDs don't match demonstration context!")
    else:
        print(f"✓ All tokens match between evaluator and demonstration.")
    print(f"{'='*180}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--hf_token", type=str, default="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")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    print("Analyzing all datasets...")
    all_results = analyze_all_datasets(tokenizer)
    print_table(tokenizer, all_results)


if __name__ == "__main__":
    main()