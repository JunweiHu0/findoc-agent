# FinDoc Agent · 金融年报多模态 Agent

你好 👋 这是基于 **ColQwen2 视觉检索 + LangGraph 状态机** 的金融文档问答系统。

## 怎么用

直接在下方输入问题，或点击 starter 卡片快速试用。

## 核心组件

- 🔍 **ColQwen2** 视觉检索：直接看页面图片，绕过 OCR
- 🧠 **DeepSeek** Planner / Verifier / Synthesizer 三段式 agent
- 👁️ **Qwen VLM** 阅读召回页面，提取数字与事实
- ↻ **Reflexion 循环**：证据不足时自动追加子任务再检索

## 你会看到什么

- 每个 agent 节点会以**可折叠 Step** 出现在回答上方
- 召回的页面以**缩略图**挂在最终回答下面，点击放大
- 答案中带 `[doc_id p.X]` 引用，可追溯到具体页面
