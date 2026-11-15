# 三条路线实测

报告时区 Asia/Shanghai，数据截至 2026-11-13。预训练骨干没有整网重训。2025–2026 是探索性再验证，不是独立最终样本外。本轮没有自动建立监控。

拟合窗在冻结规则下只有 0 笔 HIT 候选。2024-07-01 至 2024-10-31 上证多数交易日被研究近似判为 DOWN，因此不开新仓。logistic 与树没有可训练标签：fit rows 0 classes 0。这不是把测试期换掉再挑赢家。
未过滤研究近似在再验证里成交 26 笔，胜率 38.5%，方向准确率 46.2%，平均净收益 0.2%，profit factor 1.35，组合最大回撤 -7.8%。笔数少于 30，不能把这个点估计写成成功。方向准确率和交易胜率不是同一个数。

## 对照

| 路线 | 候选 | 再验证笔数 | 胜率 | 相对对照 | 区块下界 | PF | 平均净收益 | 判定 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 HIT过滤 | hit_kronos_pos | 11 | 36.4% | -2.1pp | -6.9pp | 1.08 | -1.0% | 无效 |
| 1 HIT过滤 | hit_kronos_cost | 11 | 36.4% | -2.1pp | -6.9pp | 1.08 | -1.0% | 无效 |
| 2 校准消融 | logit_c1_kronos | 0 | NA | NA | NA | NA | NA | 未完成 |
| 2 校准消融 | logit_c01_kronos | 0 | NA | NA | NA | NA | NA | 未完成 |
| 2 校准消融 | tree_d2_kronos | 0 | NA | NA | NA | NA | NA | 未完成 |
| 3 最后层 | hit_finetune_pos | 11 | 36.4% | 0.0pp | 0.0pp | 1.08 | -1.0% | 无效 |

## 机会损失

下面是未过滤组合真正成交、但被 Kronos 规则挡掉的单子。赢家损失是这些被挡赢家的扣费净利合计。

| 窗口 | 规则 | 挡掉 | 其中赢家 | 赢家净利合计 | 挡掉的亏损笔 |
| --- | --- | ---: | ---: | ---: | ---: |
| fit | hit_kronos_pos | 0 | 0 | 0 | 0 |
| fit | hit_kronos_cost | 0 | 0 | 0 | 0 |
| fit | hit_finetune_pos | 0 | 0 | 0 | 0 |
| check | hit_kronos_pos | 0 | 0 | 0 | 0 |
| check | hit_kronos_cost | 0 | 0 | 0 | 0 |
| check | hit_finetune_pos | 0 | 0 | 0 | 0 |
| reval | hit_kronos_pos | 16 | 7 | 59671 | 9 |
| reval | hit_kronos_cost | 16 | 7 | 59671 | 9 |
| reval | hit_finetune_pos | 16 | 7 | 59671 | 9 |

## 微调

实际更新的是 `DualHead`，可训练参数 1050624 / 24741376。步数 20，学习率 0.0001，序列 64，批大小 4，用时 0.8 秒。目标是官方同一形式的 s1/s2 交叉熵，骨干冻结且前向用 eval 以关掉 dropout。权重文件 `/workspace/experiments/ashare_kronos/routes3/outputs/dual_head_finetune.pt`。这不是分类器冒充的权重更新，也不是全模型重训成功。

官方全量脚本没有跑起来：

- 官方训练依赖 comet_ml 不可用：No module named 'comet_ml'
- 官方训练依赖 qlib 不可用：No module named 'qlib'
- cuda=False；finetune/train_predictor.py 需要 torchrun、DDP 和 QlibDataset，本机不启动该全量脚本。

## 结论

三条都跑完了各自能跑的部分。HIT 研究近似可以离线复现，但它不是仓库里的实盘 HIT。Kronos 过滤和 20 步最后层微调都没有提高再验证胜率。logistic 因拟合窗没有样本而没有训练，不能声称校准模型已测出 Kronos 增量。

- 1 HIT过滤 / hit_kronos_pos：无效。
- 1 HIT过滤 / hit_kronos_cost：无效。
- 2 校准消融 / logit_c1_kronos：未完成。
- 2 校准消融 / logit_c01_kronos：未完成。
- 2 校准消融 / tree_d2_kronos：未完成。
- 3 最后层 / hit_finetune_pos：无效。

没有一条同时满足再验证的笔数、覆盖率、正期望和胜率差下界。因此本轮没有找到可实施的胜率提升。Kronos 的增量没有被这些对照证实。下一步不做全模型训练，也不自动建监控；若还要验证，需要 2026-11-13 之后的纸面记录。

方向准确率按信号日后第 5 个收盘的涨跌计算，交易胜率按扣费净利是否为正计算，两者分开列在 `all_scores.csv`。


## 补充事实

未过滤组合被 Kronos 挡掉的 16 笔，赢家净利合计约 59671 元，亏损笔净利合计约 -35561 元，挡掉交易的净利仍约为 +24109 元。过滤器丢掉的是净贡献为正的一组交易。

零样本预测和 20 步 DualHead 微调后的三日收益相关约 0.992，39 个信号里符号翻转 0 次，所以 `hit_kronos_pos`、`hit_kronos_cost` 和 `hit_finetune_pos` 的成交名单相同。费用滑点加倍后，未过滤胜率仍是 38.5%，平均净收益降到约 0.06%；过滤后的平均净收益约为 -1.2%。

再验证未过滤成交按年：2025 年 23 笔、胜率约 43.5%；2026 年只有 3 笔且全部亏损。样本太小，不能外推。

池子是 2026-11-15 点名的现存流动性股票，不是历史成份，有存活偏差。MACD 柱、成交量、止损和上证状态都是本轮冻结的研究近似。
