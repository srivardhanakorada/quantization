import matplotlib.pyplot as plt

# Data
alphas = [1.0, 0.9, 0.75, 0.5, 0.25]
fp_gcd = [0.010, 0.030, 0.191, 0.938, 1.000]
q_gcd  = [0.020, 0.036, 0.215, 0.927, 0.979]

# Plot
plt.figure(figsize=(6.5, 4.5))

plt.plot(alphas, fp_gcd, marker='o', linewidth=2.2, markersize=7, label='Full precision')
plt.plot(alphas, q_gcd,  marker='s', linewidth=2.2, markersize=7, linestyle='--', label='Quantized')

# Annotate points
# for x, y in zip(alphas, fp_gcd):
#     plt.text(x, y + 0.035, f"{y:.3f}", ha='center', va='bottom', fontsize=9)

# for x, y in zip(alphas, q_gcd):
#     plt.text(x, y - 0.065, f"{y:.3f}", ha='center', va='top', fontsize=9)

# Axis styling
plt.xlabel(r'Update scaling factor $\alpha$', fontsize=12)
plt.ylabel('Erasure failure rate', fontsize=12)
plt.title('Effect of Update Scale on Erasure', fontsize=13)

plt.ylim(-0.02, 1.08)
plt.xticks(alphas, [str(a) for a in alphas], fontsize=10)
plt.yticks(fontsize=10)

# Show alpha decreasing from left to right
plt.gca().invert_xaxis()

# plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(fontsize=10, frameon=True)
plt.tight_layout()

# Save
plt.savefig("update_scaling_curve.png", dpi=300, bbox_inches="tight")