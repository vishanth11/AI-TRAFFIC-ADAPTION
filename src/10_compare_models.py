import numpy as np
import matplotlib.pyplot as plt

# Model names
models = ["Random Forest", "GRU", "GNN", "ST-GNN"]

# Results obtained from experiments
mae = [0.031496, 0.048120, 0.063439, 0.063466]
rmse = [0.046954, 0.066062, 0.087134, 0.087511]
r2 = [0.914878, 0.838944, 0.605754, 0.553299]

# ============================================================
# MAE
# ============================================================

plt.figure(figsize=(9, 6))
bars = plt.bar(models, mae)

plt.ylabel("MAE")
plt.title("Model Comparison - Mean Absolute Error")
plt.xticks(rotation=15)

for bar, value in zip(bars, mae):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(
    r"D:\traffic_prediction\results\model_comparison_mae.png",
    dpi=300
)
plt.show()


# ============================================================
# RMSE
# ============================================================

plt.figure(figsize=(9, 6))
bars = plt.bar(models, rmse)

plt.ylabel("RMSE")
plt.title("Model Comparison - Root Mean Squared Error")
plt.xticks(rotation=15)

for bar, value in zip(bars, rmse):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(
    r"D:\traffic_prediction\results\model_comparison_rmse.png",
    dpi=300
)
plt.show()


# ============================================================
# R²
# ============================================================

plt.figure(figsize=(9, 6))
bars = plt.bar(models, r2)

plt.ylabel("R² Score")
plt.title("Model Comparison - R² Score")
plt.xticks(rotation=15)

for bar, value in zip(bars, r2):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(
    r"D:\traffic_prediction\results\model_comparison_r2.png",
    dpi=300
)
plt.show()


print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

for i, model in enumerate(models):
    print(
        f"{model:20s} "
        f"MAE={mae[i]:.4f}  "
        f"RMSE={rmse[i]:.4f}  "
        f"R²={r2[i]:.4f}"
    )

print("=" * 60)