


# 使用方法

## 配置参数
在 `config-non-homogeneous-models.json5` 文件中设置任务、模型路径、语言等参数。

## 运行脚本
在命令行中运行 `CAL-Integration-non-homogeneous.py` 脚本：
```bash
python CAL-Integration-non-homogeneous.py
```

## 查看结果
脚本运行完成后，结果将保存在指定的文件中。

---

# 代码结构

## 主函数
- **main函数**：负责加载模型、读取数据、计算相似性并保存结果。

## 相似性度量函数
- **cal_RSM函数**：计算 RSM 相关的相似性度量。
- **cal_Topology函数**：计算拓扑相关的相似性度量。
- **cal_Alignment函数**：计算对齐相关的相似性度量。
- **cal_Statistic函数**：计算统计相关的相似性度量。
- **cal_Neighbors函数**：计算近邻相关的相似性度量。

## 辅助函数
如以下函数用于具体的相似性度量计算：
- `calculate_rsm_norm_difference`
- `calculate_rsa`
- `calculate_cka`

---

# 示例

运行以下命令：
```bash
python CAL-Integration-non-homogeneous.py
```

---

# 注意事项

1. 确保所有依赖库已正确安装。
2. 配置文件中的路径和参数应根据实际情况进行调整。
3. 如果遇到性能问题，可以考虑：
    - 使用 GPU 加速。
    - 优化代码逻辑。
