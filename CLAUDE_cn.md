# CLAUDE_cn.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 项目范围

基于 MMFF94 的环张力能量量化，**仅适用于单环饱和碳环**。多环（桥环/稠环/螺环）、杂环、芳香环或不饱和环的输入将被拒绝，返回 `is_supported=False` 和"Not Supported"消息，不要将分析流程扩展到其他环系。

预期用途是**可合成的真实世界环**：预测张力（包括空间位阻/vdW 贡献）并区分立体异构体（顺/反），适用于可能在也可能不在参考数据库中的化合物。通用性通过对精选文献基准点按环大小进行线性标定，以及完全绕过标定的力场原生立体异构体能量差来实现。

## 已知局限：极端拥挤

当每个环碳原子都带有多个非氢取代基时，**全取代小环**的同源脱环参比反应会变得不符合物理实际。在梯度压力测试中观察到的例子：

- 1,1,2,2-四乙基环丙烷：模型值 +28.16 kcal/mol——标定后的张力排序与甲基类似物相反。
- 六甲基环丙烷：模型值 +27.41 kcal/mol——尽管拥挤程度更严重，但定性上低于四叔丁基环丙烷。
- 六乙基环丙烷：**原始 MMFF 张力 −10.62 kcal/mol**，全部四个能量分解项（vdW、扭转、角度、键长）均为负值。标定强行将其推回 +12.65，但该数值没有物理意义。

根本原因：在开链参比中，每个甲基/乙基对贡献的 vdW 排斥力*同样*存在于环中；`环状 − 开链` 的差值因此抵消了本应暴露的张力。当原始能量分解本身符号翻转时，按环大小的线性标定器无法恢复有意义的张力值。

这些化合物充其量只是转瞬即逝的实验室奇观（据所查文献，六乙基环丙烷尚未被合成），因此该失效区域被记录而非修复。**不要尝试将分析扩展到多取代小环，除非将参比反应改为严格的键平衡方案**（这是 `plans/humming-wobbling-porcupine.md` 中的路径 2，且是有意未选择的）。

## 运行

所有命令从此目录（`C:\Users\miaoc\Desktop\mmff94`）运行。代码库中的导入均为顶层导入（`from ring_strain.core import ...`），因此项目根目录必须在 `sys.path` 上——当从此处通过 `python -m ...` 调用时，它会自动满足此条件。

```bash
# 单个 SMILES（环丙烷）
python -m cli.main "C1CC1"

# JSON 输出，并与参考值比较
python -m cli.main "C1CC1" --json --compare-ref

# 从文件批量处理（每行一个 SMILES，支持 `#` 注释）
python -m cli.main --batch compounds.txt

# 参考数据库 / 与实验值基准对比
python -m cli.main --list-refs
python -m cli.main --benchmark

# 重新生成按环大小的标定系数
# （写入 ring_strain/calibration_coefficients.json；约 10-15 分钟）
python scripts/derive_calibration.py
```

会显著影响结果的 CLI 参数：`--n-conformers`（默认 200）、`--mc-steps`（默认 500）、`--seed`（默认 42）、`--threshold`（默认 8.0）、`--no-monte-carlo`（禁用取代环的 MC/PT——见下文"构象采样约定"）。

## 测试

```bash
python -m pytest tests/                                  # 完整测试套件（约 5 分钟）
python -m pytest tests/test_regression.py                # 核心环烷烃回归
python -m pytest tests/test_bulky_regression.py          # 叔丁基/偕二甲基接线测试（快速）
RUN_BENCHMARKS=1 python -m pytest tests/test_bulky_benchmark.py   # 8 个化合物的大位阻精度基准（约 7 分钟）
python -m pytest tests/test_regression.py::TestCycloalkaneStrain::test_cycloalkane_strain
```

`tests/test_bulky_benchmark.py` 默认被跳过；设置 `RUN_BENCHMARKS=1` 来运行。核心环烷烃回归使用 ±5 kcal/mol 的容差（与文献对比）；大位阻基准使用 ±2.5 kcal/mol。

## 脚本

```bash
# 推导：从参考集合重新生成标定系数（约 10-15 分钟）
python scripts/derive_calibration.py

# 验证：留出法泛化——8 个不在参考数据库中的真实世界化合物
python scripts/test_heldout_generalization.py

# 验证：1,2/1,3/1,4-二甲基环己烷的顺/反环能量差
python scripts/verify_stereo_gap.py

# 验证：真实世界单取代环张力精度 + 立体异构体能量差
python scripts/validate_real_world.py

# 诊断：追踪顺/反对的同源脱环反应
python scripts/diag_homodesmotic.py
```

## 示例

`examples/basic_usage.py` 演示了 API：单分子分析、环烷烃系列、空间位阻效应、不支持的输入以及成对比较。使用 `python examples/basic_usage.py` 运行。

## 架构

`ring_strain/core.py` 中的 `StrainAnalyzer.analyze(smiles)` 是唯一入口点；其余所有组件都是它实例化的协作者。流程如下：

1. **解析 + 立体异构体枚举**（`core._maybe_enumerate_stereo`）——如果 SMILES 有 ≥2 个未指定的环立体中心（例如没有 `@` 标记的 1,3-二叔丁基环己烷），则调用 `EnumerateStereoisomers`，对每个规范异构体递归调用 `analyze`，返回最低张力报告，并附带 `strain_range_kcal_mol = (min, max)` 和 `stereoisomers_analyzed`。由于枚举后的异构体立体化学已完全指定，递归会终止。报告还额外暴露 `stereoisomer_breakdown`（每个异构体的 SMILES + 标定值 + 原始值 + 环能量）和 `stereoisomer_cyclic_gap_kcal_mol`（所有非对映异构体之间原始 MMFF 环能量的最大值 − 最小值）。**环能量差是力场原生的顺/反区分指标**——它绕过了开链参比和按环大小的标定，当某环大小的标定斜率远小于 1 时，这两者都可能压缩小的异构体差异。在报告立体异构体能量差时使用此字段；在绝对张力比较时使用 `total_strain_calibrated_kcal_mol`。
2. **环验证**（`ring_analysis.RingAnalyzer.validate`）——拒绝所有不属于单环饱和碳环的输入。
3. **环状构象搜索**（`mmff.MMFFCalculator`）——ETKDG 嵌入 + MMFF94 并行优化所有构象，构象数量根据可旋转键和 `compute_bulk_score(mol)` 自适应缩放（见下文采样约定）。对于取代的 4–7 元环，`seed_ring_pucker_conformers` 随后添加额外的随机坐标 ETKDG 种子（`useRandomCoords=True`，`clearConfs=False`），以确保椅式/扭船式/信封式/半椅式构象盆地都被采样。随后进行 PT + MC 扭转搜索，步数同样根据体积缩放。当 `use_boltzmann=True` 时，对存活的构象进行聚类 + 玻尔兹曼平均。
4. **张力计算**（`homodesmotic.HomodesmoticAnalyzer.compute_strain`）——两种模式：
   - **未取代环烷烃**：严格的键平衡反应 `环-(CH₂)ₙ + CH₃CH₃ → CH₃(CH₂)ₙ₊₁CH₃`。每个环大小的张力缓存在 `_CYCLOALKANE_STRAIN_CACHE` 中。
   - **取代环**：在每个环单键处开环，通过规范 SMILES 去重，使对称环不会为等效的开环位置付出重复计算代价。每个唯一候选结构运行完整的嵌入 + MMFF +（体积缩放的）PT/MC + 玻尔兹曼流程；保留能量最低的开链参比。
5. **各能量项分解**——`MMFFCalculator.decompose_energy` 通过切换 `SetMMFFXxxTerm(False)` 来提取环状和开链结构的 `{bond, angle, stretch_bend, oop, torsion, vdw, electrostatic}`；环状减开链的差值填充 `StrainReport` 上的 `vdw_strain_kcal_mol / torsion_strain_kcal_mol / angle_strain_kcal_mol / bond_strain_kcal_mol`。**vdW 差值是有物理意义的"真实空间位阻"张力**，也是现在 `steric_confinement_kcal_mol` 报告的值。
6. **标定**（`calibrate.StrainCalibrator`）——按环大小的线性校正 `calibrated = a[size] * raw + b[size]`，从 `ring_strain/calibration_coefficients.json` 加载以实现即时初始化。对于未取代参比定义了无张力锚点的环大小（尤其是环己烷 = 0），拟合被约束为精确通过 `(raw_anchor, 0)`。只有一个参比的环大小回退到历史乘法形式。在编辑参考集合后如需重新生成，运行 `python scripts/derive_calibration.py`。
7. **评分 / 参考匹配**——`scoring.StabilityScorer` 将标定后的张力映射为 0–100 分（默认阈值 8 kcal/mol）；`reference.ReferenceDatabase` 按规范 SMILES 查询。

几何诊断（`geometry.GeometryAnalyzer`）与能量路径并行运行，在报告中贡献 Baeyer（角度 RMS）、Pitzer（重叠扭转计数）和跨环接触字段——它们来自单个最佳构象的诊断快照，**当玻尔兹曼平均处于活跃状态时，不能保证各项之和等于 `total_strain_mmff_kcal_mol`**。同样的注意事项适用于各能量项分解。

## 构象采样约定（关键）

对于**取代**环，`mmff.monte_carlo_search` / `parallel_tempering_search` 中的蒙特卡洛 + 并行退火扭转搜索必须保留在关键路径上。单构象 ETKDG 会产生化学上错误的结果（例如叔丁基环丙烷报告与环丙烷相同的张力，因为取代基旋转异构体从未被探索）。在修复该路径中的 bug 时：

- 只修复出错的代码行；不要通过删除 MC/PT 调用来"简化"。
- `parallel_tempering_search → cluster_conformers → compute_boltzmann_energy` 是 `homodesmotic._compute_acyclic_energy_with_mol` 内部的已验证链。新需求应添加旁路开关，而非替换默认行为。
- 构象 ID 不保证是连续的——使用 `[c.GetId() for c in mol.GetConformers()]` 而非 `range(mol.GetNumConformers())` 来迭代，否则在先前的 PT 过程移除并重新添加构象后，RDKit 会抛出"Bad Conformer Id"。
- 如果取代环报告 `~0 kcal/mol` 的张力，应先怀疑采样而非力场。

步数和 PT 温度阶梯根据 `compute_bulk_score(mol)` 缩放，该函数计算非环重原子上的 `Σ max(1, heavy_degree − 1)²`，上限为 15。环状侧使用 `pt_steps = max(100, mc_steps // 4) + 12 * bulk`，`mc_steps = mc_steps_arg + 18 * bulk`；开链侧使用 `pt_steps = 150 + 12 * bulk`，`mc_steps = 200 + 18 * bulk`——两者足够接近，使得两侧都不会获得会偏向 `cyclic_E − acyclic_E` 的系统性采样优势。PT 以四个副本运行，温度分别为 (300, 500, 1000, 2000) K。初始 ETKDG 构象数量也是自适应的：`30 + 20 * 可旋转键数 + 15 * bulk`，限制在 `[20, max(n_conformers, 1500)]`。MC/PT 步骤中，在每次 MMFF 调用之前使用 `_VDW_CLASH_THRESHOLD_A = 1.3` Å 进行重原子预筛选，拒绝将非键合原子推入共价键范围的移动。

## 标定系数

`ring_strain/calibration_coefficients.json` 由 `scripts/derive_calibration.py` 生成并提交到仓库中，使用户侧的初始化时间达到亚毫秒级。JSON 包含每个环大小的拟合 `(a, b)` 以及用于拟合它们的原始 `(raw, exp)` 数据点。每当 `ring_strain/reference.py` 发生变化时重新生成它。该脚本需要约 10–15 分钟，因为它对每个参考化合物运行完整的生产流程（包括 14 个取代化合物，每个耗时 30–90 秒）。

`StrainCalibrator` 加载顺序：
1. 本进程中先前调用的类级缓存。
2. JSON 文件（生产路径）。
3. 仅从**未取代**参比实时推导——快速（约 1 秒），但仅产生每个环大小的乘法因子。这是 JSON 缺失或损坏时的回退方案；它有意跳过取代参比，因为它们每个都需要几分钟。

## 本仓库特定约定

- `StrainReport` 同时携带原始 MMFF94 张力（`total_strain_mmff_kcal_mol`）和标定后的值（`total_strain_calibrated_kcal_mol`）——在与文献比较时，始终使用标定后的值。
- `MMFFGetMoleculeForceField` 调用在 `mmff.py` 中被包装在 `_suppress_cpp_stderr()` 中，因为 RDKit 的 C++ BFGS 优化器直接将"Invariant Violation"消息写入文件描述符 2，绕过了 Python 日志。
- 能量统一使用 kcal/mol；角度使用度数；距离使用 Ångström。
- `ring_strain/calibration_coefficients.json` 是生成的文件，不要手动编辑——每当参考集发生变化时，通过 `scripts/derive_calibration.py` 重新生成。
