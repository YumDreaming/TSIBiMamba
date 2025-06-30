import matplotlib.pyplot as plt
import math

# 数据准备
SEQ_LENGTHS = [500, 1000, 2000, 2500, 3000, 4000,
               5000, 6000, 7000, 8000, 10000, 13000, 15000]
mem_lists = [
    [0.607, 1.289, 2.315, 3.311, 3.972, 4.988, 6.333, 7.668, 9.006, 10.352, 12.381, 15.768, 19.094],
    [1.022, 1.722, 2.831, 3.960, 4.714, 5.841, 7.348, 8.833, 10.316, 11.818, 14.083, 17.823, 21.546],
    [1.612, 2.125, 3.865, 6.134, 8.308, 12.617, 19.490, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan],
    [1.619, 2.780, 4.982, 7.452, 9.455, 12.771, 17.862, 23.698, math.nan, math.nan, math.nan, math.nan, math.nan],
    [1.886, 2.728, 3.982, 5.231, 6.052, 7.277, 8.925, 10.585, 12.251, 13.923, 16.413, 20.559, math.nan]
]
sp_lists = [
    [0.003835, 0.006348, 0.012867, 0.016908, 0.019488, 0.025906, 0.033074, 0.039373, 0.046576, 0.052690, 0.066117, 0.089100, 0.102724],
    [0.007585, 0.009286, 0.019946, 0.029209, 0.035229, 0.050632, 0.070350, 0.091922, 0.116235, 0.140974, 0.204734, 0.318022, 0.407224],
    [0.006743, 0.004468, 0.008350, 0.013422, 0.017861, 0.029442, 0.047872, 0.066060, math.nan, math.nan, math.nan, math.nan, math.nan],
    [0.007937, 0.010547, 0.021989, 0.033451, 0.039931, 0.054043, 0.080976, 0.097290, 0.132293, 0.150220, math.nan, math.nan, math.nan],
    [0.006560, 0.012168, 0.027805, 0.039635, 0.049885, 0.072341, 0.101875, 0.132317, 0.169061, 0.205960, 0.298021, 0.467923, 0.597393]
]
model_names = ["TSIBiMamba", "TSIBiTransformer", "MAET", "Conformer", "ViTransformer"]
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#8C8C8C"]

# GPU 显存对比图
fig, ax = plt.subplots(figsize=(7, 4), dpi=900)
for y, lab, c in zip(mem_lists, model_names, colors):
    ax.plot(SEQ_LENGTHS, y, marker='o', markersize=4, linewidth=1.4, color=c, label=lab)
# 外推虚线
y_top = ax.get_ylim()[1]
for idx in (2, 3, 4):
    y = mem_lists[idx]
    for i, v in enumerate(y):
        if math.isnan(v) and i >= 2:
            x1, y1 = SEQ_LENGTHS[i-1], y[i-1]
            x2, y2 = SEQ_LENGTHS[i-2], y[i-2]
            slope = max((y1 - y2)/(x1 - x2),1e-6)*1.5
            next_x = SEQ_LENGTHS[i] if i<len(SEQ_LENGTHS) else x1*1.1
            x_end = x1 + (y_top - y1)/slope
            if not (x1 < x_end < next_x):
                x_end = x1 + (next_x - x1)*0.999
            ax.plot([x1, x_end], [y1, y_top], linestyle='--', linewidth=1.4, dashes=(6, 4), color=colors[idx])
            break
ax.set_xlabel("Sequence Length", fontsize=17, )
ax.set_ylabel("GPU Memory (GB)", fontsize=17, )
ax.tick_params(axis='both', labelsize=12)
# 压缩图例框
ax.legend(loc='upper left', fontsize=14, frameon=True,
          handlelength=0.9, handletextpad=0.2, columnspacing=0.2, borderpad=0.2,
          framealpha=0,  # 背景透明
          edgecolor='none'  # 无边框
          )
fig.tight_layout()
fig.savefig("C:/Users\Yum\Desktop\cpp\pic01\memory_plot.pdf", format='pdf', bbox_inches='tight')

# 推理速度对比图
fig, ax = plt.subplots(figsize=(7, 4), dpi=900)
for y, lab, c in zip(sp_lists, model_names, colors):
    ax.plot(SEQ_LENGTHS, y, marker='o', markersize=4, linewidth=1.4, color=c, label=lab)
# 外推虚线
y_top = ax.get_ylim()[1]
for idx in (2, 3, 4):
    y = sp_lists[idx]
    for i, v in enumerate(y):
        if math.isnan(v) and i >= 2:
            x1, y1 = SEQ_LENGTHS[i-1], y[i-1]
            x2, y2 = SEQ_LENGTHS[i-2], y[i-2]
            slope = max((y1 - y2)/(x1 - x2),1e-6)*1.5
            next_x = SEQ_LENGTHS[i] if i<len(SEQ_LENGTHS) else x1*1.1
            x_end = x1 + (y_top - y1)/slope
            if not (x1 < x_end < next_x):
                x_end = x1 + (next_x - x1)*0.999
            ax.plot([x1, x_end], [y1, y_top], linestyle='--', linewidth=1.4, dashes=(6, 4), color=colors[idx])
            break
ax.set_xlabel("Sequence Length", fontsize=17, )
ax.set_ylabel("Inference Time (s)", fontsize=17, )
ax.tick_params(axis='both', labelsize=12)
ax.legend(loc='upper left', fontsize=14, frameon=True,
          handlelength=1.0, handletextpad=0.3, columnspacing=0.3, borderpad=0.2,
          framealpha=0,  # 背景透明
          edgecolor='none'  # 无边框
          )
fig.tight_layout()
fig.savefig("C:/Users\Yum\Desktop\cpp\pic01\speed_plot.pdf", format='pdf', bbox_inches='tight')

plt.show()
