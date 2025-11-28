import json
import csv
import numpy as np
import os
import pandas as pd

def main(json_path, dataset):
    csv_path = f"{json_path}/{dataset}_acc_f1_kl_results.csv"

    with open(f"{json_path}/result_dict.json", "r") as f:
        data = json.load(f)
    test_result = data["test_result"]

    # KL data
    kl_path = os.path.join(json_path, "kl_divergence.json")
    kl_data = None
    if os.path.exists(kl_path):
        with open(kl_path, "r") as f:
            kl_data = json.load(f)

    rows_zf_m1 = []
    rows_m2 = []
    rows_m2_adaptive = []
    best_m2 = None
    best_m2_adaptive = None

    def get_kl_stats(method):
        if kl_data is None or method not in kl_data:
            return "-", "-"
        kls = [
            v.get("mean_kl") for k, v in kl_data[method].items()
            if v.get("mean_kl") is not None
        ]
        if len(kls) == 0: return "-", "-"
        return float(np.mean(kls)), float(np.std(kls, ddof=1))

    # Zero/Few/M1
    for method in ["zero_shot", "few_shot", "m1"]:
        if method not in test_result:
            continue

        accs = [run["acc"] for run in test_result[method]]
        f1s = [run["macro_f1"] for run in test_result[method] if "macro_f1" in run]

        mean_acc = float(np.mean(accs))
        var_acc = float(np.var(accs, ddof=1)) if len(accs) > 1 else 0.0

        mean_f1 = float(np.mean(f1s)) if f1s else "-"
        var_f1 = float(np.var(f1s, ddof=1)) if len(f1s) > 1 else "-"

        mean_kl, var_kl = get_kl_stats(method)

        rows_zf_m1.append([method, "-", "-", mean_acc, var_acc, mean_f1, var_f1, mean_kl, var_kl])

    # M2 / M2-Adaptive calc
    def analyze_method(method, rows_store, current_best):
        stats = {}

        for run in test_result[method]:
            for q, lambdas in run.items():
                q_n = int(q.split("_")[0])
                for lam, metrics in lambdas.items():
                    lam_v = float(lam.split("_")[-1])
                    key = (q_n, lam_v)

                    if key not in stats:
                        stats[key] = {"acc": [], "macro_f1": []}

                    stats[key]["acc"].extend(metrics.get("accs", [metrics["acc"]]))
                    if "macro_f1" in metrics:
                        stats[key]["macro_f1"].append(metrics["macro_f1"])

        mean_kl, var_kl = get_kl_stats(method)

        q_dict = {}
        lam_dict = {}

        for (q_n, lam), vals in stats.items():
            accs = vals["acc"]
            f1s = vals["macro_f1"]

            mean_acc = float(np.mean(accs))
            var_acc = float(np.var(accs, ddof=1)) if len(accs) > 1 else 0.0
            mean_f1 = float(np.mean(f1s)) if f1s else "-"
            var_f1 = float(np.var(f1s, ddof=1)) if len(f1s) > 1 else "-"

            row = [method, q_n, lam, mean_acc, var_acc, mean_f1, var_f1, mean_kl, var_kl]
            rows_store.append(row)

            if current_best is None or mean_acc > current_best[3]:
                current_best = row

            q_dict.setdefault(q_n, []).append(mean_acc)
            lam_dict.setdefault(lam, []).append(mean_acc)

        def delta(d):
            if len(d) < 2:
                return "-"
            ks = sorted(d.keys())
            return float(np.mean(d[ks[-1]]) - np.mean(d[ks[0]]))

        return delta(q_dict), delta(lam_dict), current_best

    # Apply
    if "m2" in test_result:
        m2_dq, m2_dl, best_m2 = analyze_method("m2", rows_m2, best_m2)
    else:
        m2_dq, m2_dl = "-", "-"
    if "m2_adaptive" in test_result:
        m2a_dq, m2a_dl, best_m2_adaptive = analyze_method("m2_adaptive", rows_m2_adaptive, best_m2_adaptive)
    else:
        m2a_dq, m2a_dl = "-", "-"

    df_m2 = pd.DataFrame(rows_m2); df_m2 = df_m2.drop_duplicates(); rows_m2 = df_m2.values.tolist()
    # Save CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "queries", "ridge_lambda",
            "mean_acc", "var_acc",
            "mean_f1", "var_f1",
            "mean_kl", "var_kl"
        ])

        writer.writerows(rows_zf_m1)
        writer.writerow([])
        if best_m2: writer.writerow(["m2_best"] + best_m2[1:])
        if best_m2_adaptive: writer.writerow(["m2_adaptive_best"] + best_m2_adaptive[1:])
        writer.writerow(["effect_queries(m2)", m2_dq, "effect_queries(m2_adaptive)", m2a_dq, "effect_lambdas(m2_adaptive)", m2a_dl])
        writer.writerow([])
        writer.writerows(rows_m2)
        writer.writerows(rows_m2_adaptive)

    print("\nCSV saved:", csv_path)

def main(json_path, dataset):
    csv_path = f"{json_path}/{dataset}_acc_f1_kl_results.csv"

    with open(f"{json_path}/result_dict.json", "r") as f:
        data = json.load(f)
    test_result = data["test_result"]

    # KL data
    kl_path = os.path.join(json_path, "kl_divergence.json")
    kl_data = None
    if os.path.exists(kl_path):
        with open(kl_path, "r") as f:
            kl_data = json.load(f)

    rows_zf_m1 = []
    rows_m2 = []
    rows_m2_adaptive = []
    best_m2 = None
    best_m2_adaptive = None

    def safe_var(x):
        return float(np.var(x, ddof=1)) if len(x) > 1 else "-"

    def get_kl_stats(method):
        if kl_data is None or method not in kl_data:
            return "-", "-"
        values = [v.get("mean_kl") for v in kl_data[method].values() if v.get("mean_kl") is not None]
        return (
            float(np.mean(values)),
            safe_var(values)
        ) if values else ("-", "-")

    # Zero_shot / Few_shot / M1
    for method in ["zero_shot", "few_shot", "m1"]:
        if method not in test_result:
            continue

        accs = [run["acc"] for run in test_result[method]]
        f1s = [run["macro_f1"] for run in test_result[method] if "macro_f1" in run]

        mean_acc = float(np.mean(accs))
        var_acc = safe_var(accs)
        mean_f1 = float(np.mean(f1s)) if f1s else "-"
        var_f1 = safe_var(f1s)
        mean_kl, var_kl = get_kl_stats(method)

        rows_zf_m1.append([method, "-", "-", mean_acc, var_acc, mean_f1, var_f1, mean_kl, var_kl])

    # 분석 함수
    def analyze_method(method, rows_store, current_best):
        stats = {}

        for run in test_result[method]:
            for q, lambdas in run.items():
                q_n = int(q.split("_")[0])
                for lam, metrics in lambdas.items():
                    # lam은 adaptive일때만 의미 있음
                    lam_v = float(lam.split("_")[-1])

                    if (q_n, lam_v) not in stats:
                        stats[(q_n, lam_v)] = {"acc": [], "macro_f1": []}

                    stats[(q_n, lam_v)]["acc"].extend(metrics.get("accs", [metrics["acc"]]))
                    if "macro_f1" in metrics:
                        stats[(q_n, lam_v)]["macro_f1"].append(metrics["macro_f1"])

        mean_kl, var_kl = get_kl_stats(method)

        q_dict = {}

        for (q_n, lam_v), vals in stats.items():
            accs = vals["acc"]
            f1s = vals["macro_f1"]

            mean_acc = float(np.mean(accs))
            var_acc = safe_var(accs)
            mean_f1 = float(np.mean(f1s)) if f1s else "-"
            var_f1 = safe_var(f1s)

            # m2는 ridge 람다 없음 → 항상 "-"
            if method == "m2":
                row = [method, q_n, "-", mean_acc, var_acc, mean_f1, var_f1, mean_kl, var_kl]
            else:
                row = [method, q_n, lam_v, mean_acc, var_acc, mean_f1, var_f1, mean_kl, var_kl]

            rows_store.append(row)

            if current_best is None or mean_acc > current_best[3]:
                current_best = row

            q_dict.setdefault(q_n, []).append(mean_acc)

        # Δ(max_query - min_query)
        if len(q_dict) >= 2:
            qs = sorted(q_dict.keys())
            delta_q = float(np.mean(q_dict[qs[-1]]) - np.mean(q_dict[qs[0]]))
        else:
            delta_q = "-"

        # lambda는 adaptive만 계산
        delta_lam = "-"
        if method == "m2_adaptive":
            lam_dict = {}
            for (q_n, lam_v), vals in stats.items():
                lam_dict.setdefault(lam_v, []).append(np.mean(vals["acc"]))
            if len(lam_dict) >= 2:
                ls = sorted(lam_dict.keys())
                delta_lam = float(np.mean(lam_dict[ls[-1]]) - np.mean(lam_dict[ls[0]]))

        return delta_q, delta_lam, current_best

    # M2 / M2-adaptive 적용
    m2_dq, _, best_m2 = analyze_method("m2", rows_m2, best_m2) if "m2" in test_result else ("-", "-", None)
    m2a_dq, m2a_dl, best_m2_adaptive = analyze_method("m2_adaptive", rows_m2_adaptive, best_m2_adaptive) if "m2_adaptive" in test_result else ("-", "-", None)

    df_m2 = pd.DataFrame(rows_m2); df_m2 = df_m2.drop_duplicates(); rows_m2 = df_m2.values.tolist()
    # Save CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "queries", "ridge_lambda",
            "mean_acc", "var_acc",
            "mean_f1", "var_f1",
            "mean_kl", "var_kl"
        ])

        writer.writerows(rows_zf_m1)
        writer.writerow([])
        if best_m2: writer.writerow(["m2_best"] + best_m2[1:])
        if best_m2_adaptive: writer.writerow(["m2_adaptive_best"] + best_m2_adaptive[1:])
        writer.writerow(["effect_queries(m2)", m2_dq, "effect_queries(m2_adaptive)", m2a_dq, "effect_lambdas(m2_adaptive)", m2a_dl])
        writer.writerow([])
        writer.writerows(rows_m2)
        writer.writerows(rows_m2_adaptive)

    print("\nCSV saved:", csv_path)

if __name__ == "__main__":
    for d in ['sst2']: #['agnews','hate_speech18','mr','sst2','sst5','trec','subj']:
        path = f"/home/jiii111/ICLTV_exp1/I2CL/exps/baseline_all_test_13/Qwen/Qwen2.5-7B/{d}"
        main(path,d)