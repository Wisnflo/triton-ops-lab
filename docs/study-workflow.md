# 技术实验执行约定

所有本地命令在本仓库使用 `uv run --frozen`；GPU 命令在 NVIDIA Linux 执行。原始数据、交付包和分析 JSON 放工作区 `storage/runs/triton/`，不提交进源码仓库。

## 冻结与独立评测

`configs/study/development-suite.json` 包含三类算子开发调优、13 个逐因素 GEMM 配置及模块对照。只有完整开发证据通过 `analysis.study.audit_study` 后，才能执行：

```sh
uv run --frozen python scripts/freeze_policy.py <development-run> --output configs/policy-v1.json
```

策略是固定的近邻形状启发式：最多三个开发邻居，候选成本在各开发形状内归一化，再按距离加权。静态路径选排名第一项；短名单路径保留前两项和默认配置，最多三项。FP16/BF16 分开拟合。特征、距离及超参数在首次独立评测前固定，策略文件包含特征源码哈希和开发 CSV 哈希。它是一项实验方法，不声称发明新的学习算法。

独立集沿用 `configs/eval.json` 的 24 项算子/模块案例。`evaluation-suite.json` 与独立会话的 `replication-suite.json` 都覆盖两种计时协议各三轮，轮换后端顺序。禁止用这些测量回填策略或删除退化案例；需要修正时必须另立版本并承认已经见过的数据。

模块按相同种子、输入和权重执行完整前向，策略逐层选参。Transformer 的 `compile_mm_compat` 明确保留外部 FP32 GEMM 兼容边界，不能称为原生 Inductor GEMM。Attention 同时比较强制 math/flash/efficient SDPA，并检查实际调度名称。

首轮正式评测中，六条 Triton Attention 记录的独立 profiler 调用没有返回内核名称；原始 CSV 中的空记录保持不变。允许用相同源码、完整环境、输入种子及原先已选配置，单独执行不计时的数值复核和 profiling；不重新搜索配置、不替换计时采样。补采必须保存原始 GPU Chrome trace、执行脚本、每次尝试以及原 CSV/metadata 的哈希，通过 `dispatch-supplement/kernel-observation-supplement.json` 明确关联。审计只接受空的 Triton Attention 观测；错误字符串、不同输入/配置/环境、缺失或篡改轨迹均不能借此通过。最多三次观测，保留首次捕获成功的结果，不允许用后续成功替换它。完成审计和原始摘要必须披露补采范围，不能称为原计时过程中捕获的轨迹。

## 安全、资源及启动成本

```sh
uv run --frozen python scripts/qualify_gpu.py all
uv run --frozen python scripts/run_study.py --suite configs/study/evaluation-suite.json
uv run --frozen python scripts/run_study.py --suite configs/study/replication-suite.json
uv run --frozen python scripts/run_study.py --suite configs/study/cost-suite.json
```

资格检查包含完整 GPU 测试的 memcheck、racecheck、initcheck（全部地址空间）、synccheck，以及三类算子的 TTIR/TTGIR/LLVM IR/PTX/cubin/SASS 和 Nsight 资源、占用率、计算、访存计数器。工具缺失、权限不足、超时、主机退出异常均不能算通过。工具观测不计入性能计时。

如完整应用的 racecheck 在厂商库内核报告告警，原运行必须保持失败状态。进一步归属核验需要保留全部原日志、完整测试通过数，并用只依赖框架的独立程序复现同名厂商内核告警。后续仅允许按完整内核名排除已经归属的厂商内核，仍执行全部 GPU 测试、检查所有项目内核；initcheck/synccheck 不沿用该排除。不能把归属核验写成“告警误报”或“厂商问题已解决”。[NVIDIA 的筛选说明](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#specifying-filters)支持精确名称筛选；该说明不证明任何具体告警是误报。

补充执行脚本、命令、原失败日志、独立复现、复验结果和硬件检查由 `qualification-combined.json` 引用。完成审计的 `--hardware` 可以接收该文件；输出必须保留 `global_racecheck_status: failed` 及排除范围，不能宣传整个框架/厂商库通过了无告警安全认证。

初始化检查不能通过只筛选项目内核来绕开框架输入生产者。已用写入/读取两个内核的对照确认：全程跟踪时无错误，只跟踪读取者时，即使物理结果正确，也会报告全局内存未初始化。分段筛选方案未通过共享内存检查，不能用于完成验收。独立 GEMM 对照中，同一 cubin 在不同版本工具下出现不同结果，只能证明该案例对工具版本敏感，不能推广为完整应用的误报结论。完整测试仍需按实际工具结果核验。

初始化报错的归属补充必须保留所有内核的插桩及全部地址空间，使用工具的报告匹配机制，不能使用初始化内核排除。`project-safety-v2` 审计仅接受来自完整框架独立复现 XML 的字面函数名、16 字节共享读取及已复现指令位置；不接受通配函数名、全局内存例外或未复现位置。必须核验规则生效的正对照、故意未初始化的全局读取，以及同类型 16 字节共享读取在全部六个项目内核名称下仍可被检出。共享故障对照还需先通过 memcheck，排除无效地址这一混淆因素。它只限定项目检查的报告例外范围，完整应用 initcheck 保持失败；不得宣称框架报错是误报或已经修复。[NVIDIA 的报告匹配机制](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#error-suppression)用于保留插桩后的报错归属诊断，文档本身不证明具体报错无害。

Colab 回收会话后，可以补充相同源码、锁文件、软件、驱动及同型号 GPU 的安全检查；各会话身份与原日志分别保留。复用的硬件采集仍需与其原资格检查的完整环境记录一致。独立性能复验的运行时身份要求不变，不合并不同会话的时延样本。

成本实验为三个预先选定的 FP16 独立案例，各后端三轮，每轮使用独立空编译缓存随后重用该缓存。冷/热指进程开始时的 Triton/Inductor 文件缓存状态；GPU 驱动缓存、进程状态和正确性预检不等同于完全冷启动。离线开发搜索成本与小型 CPU 拟合成本分开记录；搜索耗时已含候选构造、首次调用及候选验证/计时，核算时不得重复相加。无稳定运行收益时不计算正向回本次数。

成本 JSON 中的逐轮回本次数仅是该轮正向耗时差的公式测算，不代表差异具有统计显著性或收益可稳定再现。跨轮收益正负不一致、接近测量波动的案例不能据此承诺回本；原始各轮差值、无收益时的 `null` 及离线成本应一并保留。

`build_colab_delivery.py --suite study-<phase>` 从干净提交生成私有可上传 notebook。下载证据会省略可重建的编译缓存，保留缓存清单、全部指标、日志以及显式 IR/SASS/profiling 产物。保存并核验下载包后，释放本实验运行时，核对有效会话数及消耗率。

远程显示导出成功不等于本地已收到文件。A100 补充执行增加了独立于普通下载的原始包检查点：在同一私人 notebook 的 `application/json` 输出中保存 ZIP 的完整 base64、字节数和 SHA256。下载带输出的 `.ipynb` 后，可恢复逐字节相同的原始 ZIP；仍须核对哈希、ZIP 完整性与安全路径，再执行原始数据审计。运行时已回收、只有成功文字和哈希而没有原始字节的尝试不得验收。导出检查点不修改任何计时或配置。

## 完成性检查

```sh
uv run --frozen python scripts/audit_completion.py \
  --development <development-run> --evaluation <evaluation-run> \
  --replication <replication-run> --cost <cost-run> \
  --hardware <qualification-run> --output <external-audit-directory>
```

审计核对原始覆盖、候选正确性、样本中位数、单图编译、实际内核、策略身份、独立运行时身份、缓存状态及工具日志。输出逐形状/精度/协议的比较 JSON，保留所有退化，不混合硬件或独立会话。计时端点记录温度、频率、功耗等状态，但并未锁定时钟，也不等于逐样本硬件跟踪。

`replication-consistency.json` 对齐两个 L4 会话的全部案例与基线，同时保留两组各三轮的配对加速比，分别标记中位数方向一致及所有轮次方向一致。它是观测结果的一致性描述，不是统计显著性检验；方向变化的案例不得省略或归入统一加速结论。

本地分析工具可以在被测版本冻结后改进，但完成审计必须记录两者提交号，并验证 `src`（分析目录除外）、实验配置、锁文件和执行脚本与被测版本一致。预先生成的远程交付包保留原提交，不能因后续图表或归属分析改动而悄悄更换 GPU 被测源码。

第二种 GPU 仅在既有 Colab 权益允许时补充，不新增购买。所有结论限于经过测量的形状、版本和协议。技术完成不包含论文、研究报告或答辩材料。
