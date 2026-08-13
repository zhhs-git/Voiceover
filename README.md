# Voiceover

Voiceover 是一个运行在局域网内的网页应用，可以将 EPUB、PDF 和 TXT
书籍转换为按章节组织的有声书，并支持角色感知的对话分析和配音分配。

本应用采用“主机中心”模式：主机负责运行网页服务、SQLite 数据库、上传的
书籍、Python 处理程序、模型调用和音频生成。其他用户只需要在同一个可信的
局域网内使用浏览器访问即可。当前没有登录系统，也没有按用户隔离书库。

## 项目来源与二次开发

本项目基于 [Audiobook Generator](https://github.com/yassinebk/audiobook-generator)
进行二次开发，并发布为 Voiceover。这是一个独立的新仓库，不是原仓库的官方
发布版本。

在保留原有桌面版/Tauri 和本地 Python 处理能力的基础上，本项目主要增加和调整了：

- 局域网网页模式，支持其他设备直接通过浏览器访问
- 主机集中管理 SQLite 数据库、书籍文件、章节脚本和生成音频
- EPUB、PDF、TXT 导入、正文预览以及按章节顺序展示
- 按书籍维护的角色注册表和跨章节角色上下文传递
- 浏览器端的章节分析、审阅、音频生成和持续播放流程

原项目的 MIT 许可证和版权声明已保留，具体见 [LICENSE](LICENSE)。

## 各部分运行位置

| 组件 | 运行位置 | 负责内容 |
| --- | --- | --- |
| React 前端 | 浏览器 | 导入、分析、审阅、生成和播放界面 |
| Python 网页服务 | 主机 | HTTP API、文件上传、共享 SQLite 数据库和静态文件 |
| Python 处理程序 | 主机 | EPUB/PDF/TXT 提取、OCR、章节分析、TTS 和音频合成 |
| SQLite 和书籍文件 | 主机 | 共享书库、脚本、上传文件和生成的音频 |

只有主机需要访问书籍文件、数据库、Python 环境、模型配置和 TTS 凭据。
局域网用户通过浏览器上传或操作，数据会由主机统一保存和处理。

## 功能

- 导入 EPUB、PDF 和 TXT 书籍，扫描版 PDF 支持 OCR 回退
- 分析前预览提取出的章节正文
- 按章节分析旁白、对话、说话人、情绪和语速
- 维护整本书范围内的角色注册表，并使用系统生成的稳定角色 ID
- 审阅角色身份、别名和声音分配
- 在浏览器中生成和播放章节音频
- 无需账号即可让可信局域网用户共同使用同一个书库

## 界面截图

![书库](docs/screenshots/library.png)

![章节分析](docs/screenshots/book-detail-analyze.png)

![审阅](docs/screenshots/book-detail-review.png)

![音频生成](docs/screenshots/book-detail-generate.png)

## 架构

```text
浏览器用户
      │ HTTP
      ▼
Python 局域网网页服务 :8000
      ├── React 生产文件
      ├── SQLite 数据库
      ├── uploads/ 和 books/<book-id>/ 工作目录
      └── Python 处理程序
            ├── EPUB/PDF/TXT 提取和 OCR
            ├── LLM 角色与对话分析
            ├── TTS 语音合成
            └── 章节音频合成
```

仓库中仍保留桌面版/Tauri 文件，用于兼容原有的桌面工作流。当前推荐的
局域网共享方式是 Python 网页服务加 React 前端。

## 端口和运行模式

网页应用有两种常用运行模式：

- `8000` 是 Python 网页服务端口。生产模式下，它还会直接提供构建后的
  React 前端。局域网其他设备访问：`http://<主机局域网 IP>:8000`。
- `5173` 是 Vite 开发前端端口，会把 `/api` 请求代理到主机的 `8000` 端口。
  开发时访问：`http://<主机局域网 IP>:5173`。

### 环境要求

- Node.js 20 或更高版本
- Python 3.12 或更高版本
- Python 依赖管理工具 `uv`
- 当前主要支持 macOS；如果已正确安装依赖，也可以使用 CPU 或其他受支持的
  PyTorch 设备

### 安装依赖

```bash
npm install
cd workers/python
uv sync
cd ../..
```

### Whisper 转录依赖（背景音/音效分析所需）

点击“分析背景音/音效”后，程序会在原章节配音完成的基础上使用本地 Whisper 生成带时间戳的转录，再由音频规划 LLM 安排背景音乐和音效。没有 Whisper 不影响书籍导入、章节分析和原章节配音，但不能执行这一阶段。

项目的 Python 依赖已声明 `mlx-whisper`；Apple Silicon 主机按上面的 `uv sync` 安装即可。默认模型是 `mlx-community/whisper-large-v3-turbo`。如果主机已经在其它 Python 环境安装了 `mlx-whisper`，程序会自动尝试复用；也可以显式指定解释器：

```bash
export AUDIOBOOK_WHISPER_PYTHON="$HOME/.hermes/hermes-agent/venv/bin/python"
```

模型如果已在 Hugging Face 本地缓存，程序会直接使用本地快照；没有缓存时，首次转录需要下载模型。Worker 请求也可以用 `whisperModel` 和 `whisperPython` 覆盖默认值。

### 生产模式：局域网使用

```bash
npm run web
```

该命令会先构建 React 前端，然后在 `0.0.0.0:8000` 启动 Python 网页服务。
启动后，可从任意可信的局域网设备访问终端中显示的主机地址。

### 开发模式

```bash
npm run web:dev
```

该命令会同时启动 Python API（`8000`）和 Vite 前端（`5173`）。如果修改了
Python 代码或提示词，可以在终端按 `Control-C` 停止服务，再重新运行该命令。

在 macOS 上，也可以创建一个可选的桌面快捷方式：

```bash
ln -s "$(git rev-parse --show-toplevel)/scripts/Audiobook-Generator.command" "$HOME/Desktop/Voiceover.command"
```

## 数据存储和隐私

默认情况下，网页服务会把共享数据保存在仓库之外：

- macOS：`~/Library/Application Support/audiobook-generator/`
- Windows：`%APPDATA%/audiobook-generator/`
- Linux：`$XDG_CONFIG_HOME/audiobook-generator/`，或 `~/.config/audiobook-generator/`

该目录包含 `audiobook.db`、上传的源文件、提取后的章节、分析脚本和生成的
音频。可以通过 `AUDIOBOOK_DATA_DIR` 指定其他主机目录。

当前局域网服务没有认证、授权和按用户隔离功能，只应暴露在可信的私有网络中。
任何能够访问网页的用户都可以查看共享书库并启动处理任务。

正文提取、数据库存储和本地处理都在主机上完成。但应用也可以配置为调用外部服务：

- 如果 OpenAI 兼容 LLM 的 `baseUrl` 指向远程服务，章节正文和分析提示词可能会
  发送到该服务
- 当前网页生成流程默认使用 Xiaomi MiMo V2.5 voice-design TTS，会向 MiMo
  服务发送语音合成请求
- Python 处理程序仍保留 Kokoro 和 Parler 后端，用于本地或其他工作流；当前浏览器
  生成辅助函数明确调用 MiMo

只有在 LLM 和 TTS 后端都配置为本地运行时，才可以称为“完全离线部署”。

## 模型配置和密钥

处理程序会从主机用户目录中查找 OpenAI 兼容模型配置：

- `~/.pi/agent/models.json`
- `~/.pi/models.json`

可以通过 `AUDIOBOOK_LLM_MODEL` 选择已配置的模型，例如：

```bash
export AUDIOBOOK_LLM_MODEL="deepseek/deepseek-v4-flash"
```

实际默认模型取自模型配置中的 `default` 项，网页界面不会硬编码默认模型。
如果找不到可用的模型配置，处理程序会使用确定性的 mock 分析器，便于测试流水线。

为兼容不同的 OpenAI 兼容网关，分析请求默认不发送 `response_format`，而是由提示词和本地 JSON 解析器约束结构化输出。只有确认某个网关支持 JSON mode 时，才在提供方或模型条目中设置
`"supportsResponseFormat": true`。

不要把 API Key 写入仓库。建议在模型配置中使用环境变量引用，而不是直接写入密钥：

```json
{
  "default": "deepseek/deepseek-v4-flash",
  "providers": {
    "deepseek": {
      "baseUrl": "https://api.deepseek.com/v1",
      "api": "openai-completions",
      "apiKeyEnv": "DEEPSEEK_API_KEY",
      "models": [
        { "id": "deepseek/deepseek-v4-flash", "maxTokens": 384000 }
      ]
    }
  }
}
```

当前 MiMo TTS 流程从主机环境变量 `MIMO_API_KEY` 或 macOS 钥匙串服务
`audiobook-generator.mimo-api-key` 中读取密钥。不要把密钥放入源代码、README、
提交到 Git 的模型配置、截图或日志中。

仓库中只应出现环境变量名称和 `test-key` 之类的测试占位值，不能包含真实凭据。

## TTS 设备配置

Kokoro 和 Parler 使用 `AUDIOBOOK_TTS_DEVICE`：

```bash
export AUDIOBOOK_TTS_DEVICE="auto"  # auto、mps、cuda 或 cpu
```

当前浏览器生成流程使用 MiMo，因此浏览器生成音频需要配置 `MIMO_API_KEY`。
设备变量只适用于基于 PyTorch 的本地 TTS 后端。

### MiMo 批量配音并发

网页端批量生成保留逐句的情绪、语速和动态演绎指导，但默认会并发两个 MiMo
片段请求以缩短等待时间。每个角色的 voice-clone 参考音频始终先串行建立或复用，
不会在并发过程中改变角色的固定音色。

```bash
export AUDIOBOOK_MIMO_CONCURRENCY="2"          # 1–4，默认 2；设为 1 可回退为串行
export AUDIOBOOK_MIMO_MAX_ATTEMPTS="3"          # 单个网络请求最多尝试次数，1–5
export AUDIOBOOK_MIMO_RETRY_BACKOFF_SECONDS="0.75"  # 重试的指数退避基准秒数
```

仅 MiMo 云端 TTS 使用此并发设置；Kokoro、Parler、Whisper 和 Stable Audio 仍按原有
本地资源限制运行。遇到瞬时并发失败时，系统只会单路重试失败片段一次，已成功片段
不会重复生成。

## 项目结构

```text
.
├── apps/desktop/          # React 前端和保留的 Tauri 源码
├── packages/shared/       # 共享 TypeScript 类型和脚本中间表示
├── workers/python/        # 网页服务和 Python 处理程序
│   ├── audiobook_worker/  # 提取、分析、TTS 和服务模块
│   └── tests/             # Python 测试
├── docs/                  # ADR、设计文档和界面截图
├── fixtures/books/        # 测试书籍存放位置
├── scripts/               # 局域网开发和 macOS 启动脚本
└── package.json           # Monorepo 脚本
```

## 测试和构建

```bash
# 前端测试
npm test --workspace @audiobook-generator/desktop -- --run

# Python 测试
cd workers/python
uv run pytest
cd ../..

# 前端生产构建
npm run build --workspace @audiobook-generator/desktop

# Rust/Tauri 检查（可选，适用于保留的桌面工作流）
cargo check --manifest-path apps/desktop/src-tauri/Cargo.toml
```

## 开发说明

- 全书角色一致性采用增量方式：选中的章节会按书籍顺序分析，每个已完成章节的
  角色注册表都会传给下一个章节
- 模型负责提出角色候选，Python 处理程序负责分配稳定的书籍级角色 ID，并持久化
  角色别名和声音分配
- 模型仍可能出现语义上的说话人判断错误，因此审阅和修正属于预期工作流的一部分

## 许可证

[MIT](LICENSE) © 2026 Voiceover contributors。
