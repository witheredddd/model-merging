# 运行与校验

## v3任务表达阶段

```bash
python scripts/tfpml/test_dense_fourier.py
bash scripts/tfpml/launch_expression.sh
```

v3默认16组频率，约27.18亿映射参数，FP32 Adam状态加参数梯度约40.5GiB，
实测训练峰值48.94GiB。训练预检需要至少150GiB可用磁盘。
所有被更新权重、数据与重建头沿用下述资产。

实际发布结果采用验证性能门槛提前结束阶段A；不是把launch脚本完整20轮跑完。
达到门槛后运行pin_expression.py锁定一个原子完成的resume状态，再停止训练进程，
随后evaluate_expression.py复核验证并唯一测试。两个脚本的默认结果路径为k16、d3配置。
结果锁定记录和变更说明见tfpml/RESULT_V3.md，不应在看过测试后换checkpoint。
pin工具以不可变硬链接保留完整状态，不会移动或修改训练器正在保存的文件。

完整映射/续训模型约31GiB，不随Git上传；模型哈希见ITERATIONS.json。

## v1/v2环境与资产

代码当前保留实际实验的服务器路径：`/home/shangxueyuan/merging/tsv`。
在其他位置运行前，需要调整tfpml脚本的root以及scripts/dtd_grassmann4.py中的ROOT；
并不是下载这个仓库后无需资产即可训练。

实际环境为Python3.10、PyTorch2.8.0+cu128、torchvision0.23.0+cu128、
open-clip-torch2.10.1、timm0.9.2、hydra-core1.3.2、wandb0.19.8、protobuf3.20.3。
全部观测包版本见tfpml/audit_results/environment_packages.json。
src包含原项目数据注册器的传递依赖；datasets、numpy、scipy、Pillow等也必须可导入。

需另外准备：

- `models/pretrained/ViT-B-16.pt`（官方OpenAI CLIP，SHA256在ITERATIONS.json）
- `models/checkpoints/ViT-B-16/pretrained.pt`（已逐张量核对的裸OpenCLIP状态）
- `models/checkpoints/ViT-B-16/head_DTDVal.rebuilt.pt`及其同名JSON来源记录
- `datasets/dtd/train`，原tanganke/dtd的3760训练图像，文件命名与原记录一致
- 若明确执行最终测试：`datasets/dtd/val`，1880测试图像

将本仓库tfpml目录放到该项目scripts/tfpml；支持源码位于src和scripts。

```bash
# 无需CLIP的完整元梯度有限差分检查
python scripts/tfpml/test_meta.py

# v2主方法：默认仅验证，不读取测试集；输出目录与v1分开
python scripts/tfpml/train.py --method D --meta-gradient full --epochs 20

# 固定选择后才允许明确安排最终测试
# python scripts/tfpml/train.py ... --final-test
```

上述v2默认配置不是已验证能够达到80%的配置；在未达到阶段B门槛时，必须报告反馈未验证。

v1原入口保存在tfpml/iterations/v1。精确复现时应把其中train.py、core.py与当前runtime.py
放入独立同级目录，再按tfpml/results/config.json的设置运行；不要把v2默认配置用于重现v1。
v1仅保留最后及冻结最佳模型，早期checkpoint不可恢复。

```bash
python verify_hashes.py
```

校验清单不包括SHA256SUMS本身或.git；Git提交哈希通过`git rev-parse HEAD`取得。
