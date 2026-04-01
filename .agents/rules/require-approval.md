---
trigger: always_on
---

# 规划与执行隔离规则 (Always-on)

当执行复杂任务或接收到开发需求时，Agent 必须严格遵守“先规划，后执行”的流程：
1. 严禁在生成规划后直接开始生成代码或修改文件。
2. 必须先输出 `Task List` 和 `Implementation Plan` 两个 Artifacts。
3. 提交 Artifacts 后，必须停止运行，并向用户发送明确提示：“请在右侧 Artifact 中添加评论以修改规划，或直接回复优化意见。确认无误后请告知‘开始执行’。”