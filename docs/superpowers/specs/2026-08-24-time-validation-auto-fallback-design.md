# 时间验证自动降级设计

## 目标

`time_validation_mode: auto` 在日期合法但无法形成完整且双类别的 chronological Train/Test/Holdout 时，自动使用现有固定种子分层随机 Train/Test；严格模式不变，非法或全空日期始终失败。

## 契约

兼容已有 `time_validation_enabled`，新增可选模式 `strict|auto|disabled`。未指定模式时，legacy true 等价 strict、false 等价 disabled。auto/strict 都请求并校验可解析日期；disabled 保持旧的非日期随机切分语义。

时间切分只有 Train、Test、Holdout 都非空且各自含 0/1 时才可用。auto 不满足该条件时复用 `_stratified_split_with_limitations()`，追加稳定限制：时间列可解析但无法形成严格 OOT 分区，已改用固定种子分层随机 Train/Test；不得声称严格 OOT、时间稳定性、生产就绪或自动上线。

## 实际执行语义

服务记录本次 `time_validation_applied`。它而非请求模式决定 validation、institution analysis、evidence time slices、report Time Decay 和 Holdout 处理。metadata/manifest 记录 mode、applied、实际 split strategy 和限制。

## 隐私与失败边界

自动回退不读取或输出行级数据，不输出路径或原始日期值。全空或非法日期维持 DataContractError，不能以随机切分掩盖数据契约问题。