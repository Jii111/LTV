import os
import csv
import json
import numpy as np
import pandas as pd


def main(json_path, dataset):

    csv_path = f"{json_path}/{dataset}_result_exp1119.csv"

    # Load result_dict.json
    with open(f"{json_path}/result_dict.json", "r") as f:
        data = json.load(f)
    test_result = data["test_result"]

    # Load KL
    kl_path = os.path.join(json_path, "kl_divergence.json")
    kl_data = None
    if os.path.exists(kl_path):
        with open(kl_path, "r") as f:
            kl_data = json.load(f)

    # Storage
    rows_zf_m1 = []
    rows_m2 = []
    rows_m2_adaptive = []
    best_m2 = None
    best_m2_adaptive = None

    # Utility
    def safe_var(x):
        return float(np.var(x, ddof=1)) if len(x) > 1 else "-"

    # ------------------------------------------------------------------
    # KL Stats: only for legacy-base methods (not m2/m2_adaptive)
    # ------------------------------------------------------------------
    def get_kl_stats(method):
        if kl_data is None or method not in kl_data:
            return "-", "-"

        vals = []
        for run_content in kl_data[method].values():
            # 예전 형식: {"mean_kl": ..., "kl_values": ...}
            if isinstance(run_content, dict) and "mean_kl" in run_content:
                vals.append(run_content["mean_kl"])

        if not vals:
            return "-", "-"

        return float(np.mean(vals)), safe_var(vals)

    # ------------------------------------------------------
    # Zero_shot / Few_shot
    # ------------------------------------------------------
    for method in ["zero_shot", "few_shot"]:
        if method not in test_result:
            continue

        accs = [run["acc"] for run in test_result[method]]
        f1s = [run["macro_f1"] for run in test_result[method]
               if "macro_f1" in run]

        mean_acc = float(np.mean(accs))
        var_acc = safe_var(accs)
        mean_f1 = float(np.mean(f1s)) if f1s else "-"
        var_f1 = safe_var(f1s)

        mean_kl, var_kl = get_kl_stats(method)

        rows_zf_m1.append([
            method, "-", "-",
            mean_acc, var_acc,
            mean_f1, var_f1,
            mean_kl, var_kl
        ])

    # ------------------------------------------------------
    # i2cl_default / i2cl_train / ICLTV
    # ------------------------------------------------------
    for method in ["i2cl_default", "i2cl_train", "ICLTV"]:
        if method not in test_result:
            continue

        accs = [run["acc"] for run in test_result[method]]
        f1s = [run["macro_f1"] for run in test_result[method]
               if "macro_f1" in run]

        mean_acc = float(np.mean(accs))
        var_acc = safe_var(accs)
        mean_f1 = float(np.mean(f1s)) if f1s else "-"
        var_f1 = safe_var(f1s)

        mean_kl, var_kl = get_kl_stats(method)

        rows_zf_m1.append([
            method, "-", "-",
            mean_acc, var_acc,
            mean_f1, var_f1,
            mean_kl, var_kl
        ])

    # ------------------------------------------------------
    # fv (float list or dict list)
    # ------------------------------------------------------
    if "fv" in test_result:
        fv_vals = test_result["fv"]

        f1s = []
        if len(fv_vals) > 0 and isinstance(fv_vals[0], dict):
            # 형태: [{"acc": ..., "macro_f1": ...}, ...]
            accs = [r["acc"] for r in fv_vals]
            f1s = [r["macro_f1"] for r in fv_vals if "macro_f1" in r]
        else:
            # 형태: [0.57, 0.57, ...]
            accs = [float(x) for x in fv_vals]

        mean_acc = float(np.mean(accs))
        var_acc = safe_var(accs)

        mean_f1 = float(np.mean(f1s)) if f1s else "-"
        var_f1 = safe_var(f1s) if f1s else "-"

        mean_kl, var_kl = get_kl_stats("fv")

        rows_zf_m1.append([
            "fv", "-", "-",
            mean_acc, var_acc,
            mean_f1, var_f1,
            mean_kl, var_kl
        ])
    # ------------------------------------------------------
    # Analyze m2 / m2_adaptive (NEW KL logic)
    # ------------------------------------------------------
    def analyze_method(method, rows_store, current_best):
        stats = {}

        # ---------- Parse test_result ------------
        for run in test_result[method]:
            for q_key, content in run.items():

                q_n = int(q_key.split("_")[0])

                # -------- m2 --------
                if method == "m2":
                    lam_v = "-"
                    if (q_n, lam_v) not in stats:
                        stats[(q_n, lam_v)] = {"acc": [], "macro_f1": []}

                    stats[(q_n, lam_v)]["acc"].append(content["acc"])
                    if "macro_f1" in content:
                        stats[(q_n, lam_v)]["macro_f1"].append(content["macro_f1"])
                    continue

                # -------- m2_adaptive --------
                for lam_key, metrics in content.items():
                    try:
                        lam_v = float(lam_key.split("_")[-1])
                    except:
                        continue

                    if (q_n, lam_v) not in stats:
                        stats[(q_n, lam_v)] = {"acc": [], "macro_f1": []}

                    stats[(q_n, lam_v)]["acc"].append(metrics["acc"])
                    if "macro_f1" in metrics:
                        stats[(q_n, lam_v)]["macro_f1"].append(metrics["macro_f1"])

        # ------------------------------------------------------
        # Build KL map for this method
        # ------------------------------------------------------
        kl_map = {}

        if kl_data is not None and method in kl_data:

            # ===== m2 KL =====
            if method == "m2":
                # structure: m2 -> run -> "32_queries" -> {"mean_kl": ...}
                for run_name, run_content in kl_data[method].items():
                    if not isinstance(run_content, dict):
                        continue
                    for q_key, q_content in run_content.items():
                        if isinstance(q_content, dict) and "mean_kl" in q_content:
                            try:
                                q_n = int(q_key.split("_")[0])
                            except:
                                continue
                            kl_map.setdefault((q_n, "-"), []).append(q_content["mean_kl"])

            # ===== m2_adaptive KL =====
            elif method == "m2_adaptive":
                # structure: run -> q_key -> lam_key -> {mean_kl}
                for run_name, run_content in kl_data[method].items():
                    if not isinstance(run_content, dict):
                        continue

                    for q_key, q_content in run_content.items():
                        if not isinstance(q_content, dict):
                            continue

                        try:
                            q_n = int(q_key.split("_")[0])
                        except:
                            continue

                        for lam_key, lam_content in q_content.items():
                            if not isinstance(lam_content, dict):
                                continue
                            if "mean_kl" not in lam_content:
                                continue

                            try:
                                lam_v = float(lam_key.split("_")[-1])
                            except:
                                continue

                            kl_map.setdefault((q_n, lam_v), []).append(lam_content["mean_kl"])

        # ------------------------------------------------------
        # Aggregate for CSV
        # ------------------------------------------------------
        q_dict = {}
        lam_dict = {}

        for (q_n, lam_v), vals in stats.items():
            accs = vals["acc"]
            f1s = vals["macro_f1"]

            mean_acc = float(np.mean(accs))
            var_acc = safe_var(accs)
            mean_f1 = float(np.mean(f1s)) if f1s else "-"
            var_f1 = safe_var(f1s)

            # (q, lambda)별 KL
            kl_vals = kl_map.get((q_n, lam_v), [])
            if kl_vals:
                mean_kl = float(np.mean(kl_vals))
                var_kl = safe_var(kl_vals)
            else:
                mean_kl, var_kl = "-", "-"

            row = [
                method, q_n, lam_v,
                mean_acc, var_acc,
                mean_f1, var_f1,
                mean_kl, var_kl
            ]
            rows_store.append(row)

            if current_best is None or mean_acc > current_best[3]:
                current_best = row

            q_dict.setdefault(q_n, []).append(mean_acc)
            if method == "m2_adaptive":
                lam_dict.setdefault(lam_v, []).append(mean_acc)

        # delta queries
        if len(q_dict) >= 2:
            qs = sorted(q_dict.keys())
            delta_q = float(np.mean(q_dict[qs[-1]]) - np.mean(q_dict[qs[0]]))
        else:
            delta_q = "-"

        # delta lambda
        if method == "m2_adaptive" and len(lam_dict) >= 2:
            ls = sorted(lam_dict.keys())
            delta_lam = float(np.mean(lam_dict[ls[-1]]) - np.mean(lam_dict[ls[0]]))
        else:
            delta_lam = "-"

        return delta_q, delta_lam, current_best

    # apply
    if "m2" in test_result:
        m2_dq, _, best_m2 = analyze_method("m2", rows_m2, best_m2)
    else:
        m2_dq = "-"

    if "m2_adaptive" in test_result:
        m2a_dq, m2a_dl, best_m2_adaptive = analyze_method(
            "m2_adaptive", rows_m2_adaptive, best_m2_adaptive)
    else:
        m2a_dq, m2a_dl = "-", "-"

    df_m2 = pd.DataFrame(rows_m2).drop_duplicates()
    rows_m2 = df_m2.values.tolist()

    # ------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "queries", "ridge_lambda",
            "mean_acc", "var_acc",
            "mean_f1", "var_f1",
            "mean_kl", "var_kl"
        ])

        writer.writerows(rows_zf_m1)

        if best_m2:
            writer.writerow(["m2_best"] + best_m2[1:])
        if best_m2_adaptive:
            writer.writerow(["m2_adaptive_best"] + best_m2_adaptive[1:])

        writer.writerows(rows_m2)
        writer.writerows(rows_m2_adaptive)

    print("\nCSV saved:", csv_path)



if __name__ == "__main__":
    for d in ['sst5','mr','trec','sst2']:
        path = f"/home/jiii111/ICLTV_exp1/I2CL/exps/baseline_all_test_15/Qwen/Qwen2.5-7B/{d}"
        main(path, d)
