# 执行说明

1. **保留旧实验目录不变。** 将 `step0_reaudit.py`, `reaudit_math.py` 放在原仓库 `dtd_schur_20261120/code/` 中，与 `bootstrap.py` 同级。
2. 需要 A100 上原 `step0_anchor0/` 目录（含 `motions.pt`, `candidate_modes.pt`, `compensation.pt` 和 JSON），以及该实验用过的 `inventory`, `probes`, `trace` 原路径。
3. 在该 `code/` 目录运行以下示例，路径替换成旧 `run_step0.py` 的实际参数：

```
python -m unittest test_reaudit_math -v
python step0_reaudit.py \
  --output /path/to/step0_anchor0 \
  --inventory /path/to/INVENTORY.json \
  --probes /path/to/probes.json \
  --trace /path/to/DTD_DYNAMIC_TRACE_20261109/trace \
  --functional-radii 0.0025 0.01
```

4. 输出仅在 `/path/to/step0_anchor0/reaudit_step0/`，原始 `CONFIG.json`, `SELECTION.json`, `.npz`, `.pt` 原封不动。
5. `patch_solver_lambda.py`（在复制出来的求解器文件上应用） 修正了新试验中 λ≠1 时的目标误差界；**当前 λ=1 不用重新跑**。不要直接修改原冻结实验的代码哈希而继续旧 campaign；应在派生代码目录测试该补丁。
6. 这份审计补丁不是新的训练算法，也未证明物理区域/流形。只用它审查已经运行完毕的 Step 0 结论。

## 注意

当前 GitHub 连接可以读取 `04047a0`，但尝试创建新分支返回 403。此补丁包尚未推送或在 A100 运行。请让具有写权限的 Codex 将文件放入独立审计分支，先检查并跑合成测试，再在服务器执行一次小半径审计。
