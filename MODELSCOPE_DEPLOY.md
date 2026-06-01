# ModelScope 部署说明（AI Data Analyst）

## 1) 创建创空间
- 进入 ModelScope 创空间，创建新空间。
- SDK 选择 `Streamlit`。
- 代码来源选择当前仓库。

## 2) 代码与依赖
- 入口文件：`app.py`
- 依赖文件：`requirements.txt`
- Python 版本建议：`3.10+`

## 3) 环境变量
在创空间的环境变量中至少配置一个：
- `OPENAI_API_KEY`
- `DEEPSEEK_API_KEY`

应用会优先读取 `OPENAI_API_KEY`，未配置时回退读取 `DEEPSEEK_API_KEY`。

## 4) 发布前自检
- 页面可打开并显示标题 `AI Data Analyst`
- `Use sample data` 可正常载入数据
- `Run AI Analysis` 可返回 summary 与图表
- 追问问题可得到回答与推荐动作
- 上传空 CSV 与异常编码 CSV 会给出错误提示

## 5) 上线后检查
- 首次冷启动后访问首页应在可接受范围内
- 中文网络访问延迟明显低于海外 Streamlit 链接
- 门户中文入口跳转到 ModelScope 链接
