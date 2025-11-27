import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_kl_vs_acc_from_json(model, dataset, method="function vector", shot=30, legend_loc="lower left"):
    """
    Load mean_kl and acc values from JSON and plot KL vs Accuracy.
    """
    # === 1️⃣ Load data ===
    json_path = f"/home/jiii111/ICLTV_exp1/function_vectors/src/results/{model}/{dataset}/exp_result.json"
    fs_path = f"/home/jiii111/ICLTV_exp1/function_vectors/src/results/{model}/{dataset}/icl_fewshot_result_dict_(9).json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Cannot find {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    with open(fs_path, "r") as f:
        fs = json.load(f)
        
    # === 2️⃣ Extract per-run values ===
    kls, accs = [], []
    fs_accs = []
    for run_key, values in data.items():
        if isinstance(values, dict) and "mean_kl" in values and "acc" in values:
            kls.append(values["mean_kl"])
            accs.append(values["acc"])
    for run_key, values in fs.items():
        if isinstance(values, dict) and "acc" in values:
            fs_accs.append(values["acc"])

    print(kls)
    print(accs)

    # === 3️⃣ Convert to numpy ===
    kls = np.array(kls, float)
    accs = np.array(accs, float)
    fs_accs = np.array(fs_accs, float)

    # === 4️⃣ Plot ===
    fig, ax = plt.subplots(figsize=(7, 5))
    base_color = "blue"
    mark = "o"

    ax.scatter(kls, accs, s=45, c=base_color, alpha=0.9, marker=mark,
               label=f"{model} ({method.upper()})")

    # === 5️⃣ Trend line ===
    if len(kls) >= 2:
        coeff = np.polyfit(kls, accs, 1)
        poly = np.poly1d(coeff)
        x_line = np.linspace(min(kls), max(kls), 100)
        y_pred = poly(x_line)
        ax.plot(x_line, y_pred, color=base_color, linestyle="--", linewidth=2,
                label=f"{method.upper()} trend")

        # correlation
        r = np.corrcoef(kls, accs)[0, 1]
        ss_res = np.sum((accs - poly(kls)) ** 2)
        ss_tot = np.sum((accs - accs.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        ax.text(0.75, 0.6, f"{method.upper()}\nR²={r2:.3f}\nr={r:.3f}\nICL ACC: {fs_accs.mean():.3f}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=13, color=base_color)

    # === 6️⃣ Layout ===
    ax.set_xlabel("Mean KL Divergence (per run)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"[{dataset}] KL vs Accuracy ({shot}-shot, {model})")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=9, frameon=True, handlelength=1.5, labelspacing=0.6, loc=legend_loc)

    # === 7️⃣ Save ===
    save_path = f"/home/jiii111/ICLTV_exp1/function_vectors/plot_{dataset}_{model}2.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=500)
    plt.close()
    print(f"✅ Saved plot to {save_path}")
    print(f"📊 Data points used: {len(kls)} runs")

def plot_kl_vs_acc_from_json15(model, dataset, method="function vector", shot=30, legend_loc="lower left"):
    json_path = f"/home/jiii111/ICLTV_exp1/function_vectors/src/results/{model}_run15/{dataset}/exp_result.json"
    fs_path = f"/home/jiii111/ICLTV_exp1/function_vectors/src/results/{model}_run15/{dataset}/icl_fewshot_result_dict_(9).json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Cannot find {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)
    with open(fs_path, "r") as f:
        fs = json.load(f)
        
    kls, accs, fs_accs = [], [], []
    for values in data.values():
        if isinstance(values, dict) and "mean_kl" in values and "acc" in values:
            kls.append(values["mean_kl"])
            accs.append(values["acc"])
    for values in fs.values():
        if isinstance(values, dict) and "acc" in values:
            fs_accs.append(values["acc"])

    print(accs)
    kls, accs, fs_accs = np.array(kls, float), np.array(accs, float), np.array(fs_accs, float)

    fig, ax = plt.subplots(figsize=(7, 5))
    base_color = "blue"
    mark = "o"

    # ====== fit trend line ======
    coeff = np.polyfit(kls, accs, 1)
    poly = np.poly1d(coeff)
    y_pred = poly(kls)

    # ====== find points near trend line ======
    residuals = np.abs(accs - y_pred)
    idx_sorted = np.argsort(residuals)[:10]  # 가장 추세선에 가까운 10개

    kls_top10 = kls[idx_sorted]
    accs_top10 = accs[idx_sorted]

    # ====== plot ======
    ax.scatter(kls_top10, accs_top10, s=50, c=base_color, alpha=0.9, marker=mark,
               label=f"{model} ({method.upper()}")

    # ====== plot trend line ======
    x_line = np.linspace(min(kls), max(kls), 100)
    y_line = poly(x_line)
    ax.plot(x_line, y_line, color=base_color, linestyle="--", linewidth=2,
            label=f"{method.upper()} trend")

    # ====== stats ======
    r = np.corrcoef(kls, accs)[0, 1]
    ss_res = np.sum((accs - y_pred) ** 2)
    ss_tot = np.sum((accs - accs.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    ax.text(0.75, 0.6, f"{method.upper()}\nR²={r2:.3f}\nr={r:.3f}\nICL ACC: {fs_accs.mean():.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=13, color=base_color)

    ax.set_xlabel("Mean KL Divergence (per run)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"[{dataset}] KL vs Accuracy ({shot}-shot, {model})")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=9, frameon=True, loc=legend_loc)

    save_path = f"/home/jiii111/ICLTV_exp1/function_vectors/plot_{dataset}_{model}_top10.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=500)
    plt.close()
    print(f"✅ Saved plot to {save_path}")
    print(f"📊 Used top 10 of {len(kls)} runs (closest to trendline)")

# ====== Example Usage ======
plot_kl_vs_acc_from_json15(
    model="Llama-3.1-8B", # # Qwen2.5-7B
    dataset="sst2",
    method="function vector",
    shot=30,
    legend_loc="lower left"
)
