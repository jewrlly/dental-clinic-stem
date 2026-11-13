"""
ЕТАП 1. Експерименти з моделлю одного каналу:
один прогін (лог, графіки), 100 прогонів (середні, гістограми), перевірка адекватності.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from params import N_RUNS
from stage1_model import simulate_one

OUT = "results"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True})


def runs_one(runs=N_RUNS, seed=0, **kw):
    rng = np.random.default_rng(seed)
    return pd.DataFrame([simulate_one(rng, **kw) for _ in range(runs)])


# ---------------- один прогін
res, log = simulate_one(np.random.default_rng(42), log=True)
log.to_csv(f"{OUT}/stage1_log.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(9, 4))
ax.step(log["Крок"], log["Черга"], where="mid", label="Довжина черги")
ax.bar(log["Крок"], log["Втрачено"], color="tab:red", alpha=.5, label="Втрачено клієнтів")
ax.set(xlabel="Крок модельного часу (робоча година)", ylabel="Кількість",
       title="Етап 1. Один стоматолог, один прогін")
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/stage1_run_queue.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(log["Крок"], log["Надійшло"], label="Надійшло клієнтів")
ax.plot(log["Крок"], log["lambda"], "k--", label="λ періоду")
ax.set(xlabel="Крок (робоча година)", ylabel="Клієнтів", title="Етап 1. Вхідний потік (один прогін)")
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/stage1_run_arrivals.png"); plt.close(fig)

# ---------------- 100 прогонів
mc = runs_one(seed=1)
mc.to_csv(f"{OUT}/stage1_100runs.csv", index=False, encoding="utf-8-sig")
mean = mc.mean(numeric_only=True)
mean.round(3).to_csv(f"{OUT}/stage1_mean.csv", header=["Середнє за 100 прогонів"],
                     encoding="utf-8-sig")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(mc["Втрачено"], bins=15, edgecolor="k")
axes[0].set(title="Втрачені клієнти (100 прогонів)", xlabel="Клієнтів", ylabel="Частота")
axes[1].hist(mc["Прибуток"], bins=15, edgecolor="k", color="tab:green")
axes[1].set(title="Прибуток (100 прогонів)", xlabel="грн", ylabel="Частота")
fig.tight_layout(); fig.savefig(f"{OUT}/stage1_100runs_hist.png"); plt.close(fig)

# ---------------- перевірка адекватності
rows = []
for ql in [0, 2, 4, 8, 16]:
    r = runs_one(seed=2, queue_limit=ql).mean(numeric_only=True)
    rows.append({"Параметр": "Місткість черги", "Значення": ql, "Втрачено": r["Втрачено"],
                 "Сер. очікування, год": r["Сер. очікування, год"],
                 "Завантаженість": r["Завантаженість"]})
for lam in [0.25, 0.5, 1.0, 2.0, 4.0]:
    r = runs_one(seed=3, const_lambda=lam).mean(numeric_only=True)
    rows.append({"Параметр": "λ (стала)", "Значення": lam, "Втрачено": r["Втрачено"],
                 "Сер. очікування, год": r["Сер. очікування, год"],
                 "Завантаженість": r["Завантаженість"]})
adeq = pd.DataFrame(rows).round(3)
adeq.to_csv(f"{OUT}/stage1_adequacy.csv", index=False, encoding="utf-8-sig")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
a = adeq[adeq["Параметр"] == "Місткість черги"]
axes[0].plot(a["Значення"], a["Втрачено"], "o-")
axes[0].set(xlabel="Місткість черги", ylabel="Втрачено клієнтів", title="Вплив місткості черги")
b = adeq[adeq["Параметр"] == "λ (стала)"]
axes[1].plot(b["Значення"], b["Завантаженість"], "o-", color="tab:purple")
axes[1].set(xlabel="λ, клієнтів/год", ylabel="Завантаженість лікаря", title="Вплив інтенсивності потоку")
fig.suptitle("Етап 1. Перевірка адекватності моделі")
fig.tight_layout(); fig.savefig(f"{OUT}/stage1_adequacy.png"); plt.close(fig)

print("Етап 1. Середнє за 100 прогонів:\n", mean.round(3))
print("\nАдекватність:\n", adeq.to_string(index=False))
