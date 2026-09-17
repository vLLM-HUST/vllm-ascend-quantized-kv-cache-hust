# 插件打包、发布与安装指南

本文介绍 vLLM-HUST Extension Bundle 插件从项目配置、构建、发布到安装使用的流程。文中使用 BidKV 作为示例：

| 项目 | BidKV 示例 |
|---|---|
| PyPI 项目名 | `bidkv` |
| Python 模块名 | `bidkv` |
| Bundle ID | `org.vllm-hust.bidkv` |
| Component ID | `victim-selector` |
| 完整组件 ID | `org.vllm-hust.bidkv/victim-selector` |
| contract | `vllm.scheduler.policy.v1` |
| execution plane | `scheduler` |

制作其他插件时，把示例中的名称换成自己项目的值。PyPI 项目名用于安装，Bundle ID 和 Component ID 用于 manifest 准入、运行时识别和组件选择。

## 1. 项目结构

一个最小的插件项目可以按下面组织：

```text
vllm-hust-bidkv/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── bidkv/
│       ├── __init__.py
│       ├── _version.py
│       ├── adapters/
│       │   └── vllm_hust/
│       │       └── selector.py
│       └── manifests/
│           └── vllm-hust-extension-v1.json
└── tests/
```

`src/bidkv/` 是 Python 包，`selector.py` 保存插件实现，manifest 描述 Bundle 及其组件，`pyproject.toml` 保存发行包元数据和构建配置。

manifest 需要打进 wheel。缺少它时，Python 包仍可能正常安装，但 vLLM-HUST 无法发现 Bundle。

## 2. 配置发行包

BidKV 使用 setuptools 构建，`pyproject.toml` 中与打包有关的配置如下：

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "bidkv"
dynamic = ["version"]
requires-python = ">=3.10"
dependencies = []

[tool.setuptools.dynamic]
version = {attr = "bidkv._version.__version__"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
bidkv = ["manifests/*.json"]

```

制作自己的插件时，需要替换项目名、Python 模块名和 Bundle ID：

```toml
[project]
name = "<PyPI 项目名>"

[tool.setuptools.dynamic]
version = {attr = "<Python 模块>._version.__version__"}

[tool.setuptools.package-data]
<Python 模块> = ["manifests/*.json"]

```

`5c994cdc` 所在的 Extension Bundle v1 实现不会扫描 Python entry point，也不会在安装后自动定位 manifest。运行时必须通过 `VLLM_EXTENSION_MANIFESTS` 显式传入 manifest 文件路径，详见第 11 节。因此，不要把自定义的 `vllm.extension_bundles` entry point 当成当前版本的发现或启用机制。

如果插件还要兼容旧版加载方式，可以额外提供 legacy entry point。BidKV 的配置是：

```toml
[project.entry-points."vllm.victim_selector"]
bidkv = "bidkv.adapters.vllm_hust.selector:BidkvVictimSelector"
```

新插件没有兼容需求时不需要这段配置。

## 3. 编写 Bundle manifest

BidKV 的 manifest 位于 `src/bidkv/manifests/vllm-hust-extension-v1.json`：

```json
{
  "schema_version": "1.0",
  "bundle_id": "org.vllm-hust.bidkv",
  "bundle_version": "0.1.0",
  "host_api_range": ">=1,<2",
  "components": [
    {
      "component_id": "victim-selector",
      "contracts": ["vllm.scheduler.policy.v1"],
      "execution_planes": ["scheduler"],
      "isolation": "trusted_in_process",
      "implementation_ref": "bidkv.adapters.vllm_hust.selector:BidkvVictimSelector",
      "permissions": []
    }
  ]
}
```

| 字段 | 说明 |
|---|---|
| `schema_version` | manifest 格式版本 |
| `bundle_id` | Bundle 的稳定标识 |
| `bundle_version` | 当前 Bundle 版本，通常与 Python 发行版本一致 |
| `host_api_range` | 支持的 vLLM-HUST host API 范围 |
| `component_id` | Bundle 内的组件名称 |
| `contracts` | 组件实现的 host contract |
| `execution_planes` | 允许加载组件的进程平面 |
| `isolation` | 组件与宿主的隔离方式 |
| `implementation_ref` | 实现类的导入路径 |
| `permissions` | 插件需要的权限 |

contract、execution plane 和 permissions 按插件的实际行为填写。比如 worker 侧插件不能直接照搬 BidKV 的 `scheduler`。

## 4. 实现组件

实现类需要满足 manifest 中声明的 contract。BidKV 的 victim selector 主要提供以下接口：

```python
class BidkvVictimSelector:
    vllm_victim_selector_api_version = 1

    @classmethod
    def from_vllm_config(cls, vllm_config):
        ...

    def pick_victim(
        self,
        running,
        policy,
        *,
        kv_utilization=None,
        now_s=None,
    ):
        ...

    def emit_observability_log(self, logger, scheduler_name):
        ...

    def export_metrics(self):
        ...
```

上述成员都是当前 `VictimSelector` runtime-checkable protocol 的结构性要求；即使观测方法不做任何事，也要实现它们。`pick_victim` 必须从非空的 `running` 候选集中返回一个 `Request`；`policy` 是 `SchedulingPolicy`，`kv_utilization` 是宿主尽力提供的 `[0, 1]` KV cache 使用率，`now_s` 是本轮调度时间戳，后两者都可能是 `None`。在该提交中，scheduler 会调用 `emit_observability_log()`，但尚没有自动汇集 `export_metrics()` 的调用点；后者仍是 protocol 兼容性的必需方法。

静态准入阶段不会加载实现模块。模块导入时不要启动线程、连接网络、访问设备或修改全局状态。配置校验放在 `from_vllm_config` 阶段；配置不合法时直接报错，不要静默降级。

## 5. 与 vLLM-HUST 主仓库适配

插件仓库独立发布，但它运行在 vLLM-HUST 的 scheduler 进程中。本节以主仓库提交 [`5c994cdc`](https://github.com/vLLM-HUST/vllm-hust/commit/5c994cdc029dfebe318ca745a39920473033038b) 为准；该提交建立了 typed scheduler-policy host seam，并保留了显式启用的 legacy entry-point 兼容路径。

### 5.1 主仓库的整体调用链

启动和调度时的数据流如下：

1. `VLLM_EXTENSION_MANIFESTS` 提供一个或多个 manifest 的绝对路径。宿主显式读取这些文件，不扫描文件系统，也不导入插件实现。
2. `resolve_extension_startup()` 先解析所有已配置 manifest，校验重复路径、Bundle ID、host API 范围、permissions 和 isolation，再应用 `VLLM_EXTENSION_BUNDLES` allowlist，生成不可变 startup snapshot。
3. scheduler 初始化时调用 `get_victim_selector()`，从 snapshot 中只查找 contract 为 `vllm.scheduler.policy.v1` 且 execution plane 包含 `scheduler` 的组件。
4. 只有一个符合条件的 provider 时自动选中；有多个时，必须用 `additional_config.victim_selector_component` 指定完整的 `<bundle_id>/<component_id>`。显式指定了不存在或未准入的组件会立即报错。
5. 选中后才按 `implementation_ref` 的 `module:attribute` 形式导入实现，并检查 API version 必须为 `1`、`from_vllm_config` 必须可调用，其返回对象必须完整实现 `VictimSelector` protocol。
6. 发生抢占时，scheduler 会排除正在进行 KV cache compression transaction 的请求，再调用 `pick_victim(candidates, policy, kv_utilization=..., now_s=...)`。本轮有抢占时，还会调用 `emit_observability_log()`。

当前宿主的 host API version 是 `1.0`。默认权限 allowlist 为空，默认只支持 `trusted_in_process`，因此 BidKV manifest 使用 `permissions: []` 和 `trusted_in_process`。这个 isolation 值不是安全沙箱：插件拥有 scheduler 进程的权限。

### 5.2 typed 与 legacy 路径的优先级

`get_victim_selector()` 的决策顺序是固定的：

1. `victim_selector_plugin_disabled=true`：直接使用 `NoOpVictimSelector`，不解析 typed provider，也不查找 legacy entry point。
2. 存在符合 contract 和 plane 的 typed provider：使用 typed 路径。一旦准入，组件选择、导入、API 版本、factory 或 protocol 检查失败都是终止错误，不会回退到 legacy。
3. 没有 typed provider，且 `additional_config.victim_selector_plugin` 显式指定了名称：在 `vllm.victim_selector` entry-point group 中精确匹配。未安装、重名、API 版本错误或 protocol 不完整都会报错。
4. 两种 provider 都没有显式启用：使用等价于 upstream 行为的 `NoOpVictimSelector`。

因此，安装 legacy entry point 不会自动启用 BidKV；必须显式配置 `victim_selector_plugin: "bidkv"`。新部署应优先使用 typed Bundle 路径，legacy 只用于旧环境兼容。

legacy 启动示例如下；使用时不要准入 typed scheduler-policy provider，否则 typed 路径优先：

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --additional-config '{"victim_selector_plugin": "bidkv"}'
```

### 5.3 版本与依赖策略

`host_api_range` 是运行时兼容性声明，不替代依赖管理。每次插件发布都应记录并测试一个明确的主仓库目标：vLLM-HUST 发布版本或 commit、Python 版本、设备/平台和所使用的 contract 版本。建议在插件仓库的 README 或发布说明中维护下表：

| 插件版本 | 已验证的 vLLM-HUST 版本或 commit | host API 范围 | contract | 验证环境 |
|---|---|---|---|---|
| `0.1.0` | `<填写实际版本或 commit>` | `>=1,<2` | `vllm.scheduler.policy.v1` | `<Python、设备、镜像>` |

若插件依赖主仓库以外的 Python 库，应写入 `[project].dependencies`。是否把 vLLM-HUST 本身列为 Python 依赖，取决于团队的部署策略：由插件安装命令负责安装宿主时应声明兼容范围；由镜像、运维环境或主仓库发行包统一提供宿主时，可不重复声明，但必须在安装说明中写明目标环境。无论哪种策略，都不能仅依赖 `dependencies = []` 而不记录实际验证的宿主版本。

### 5.4 开发与验证流程

1. 在目标 vLLM-HUST checkout 或发行环境中核对 `vllm/plugins/contracts.py`、`vllm/plugins/startup.py` 和 `vllm/v1/core/sched/victim_selector.py`；优先依赖公开 contract，避免让 BidKV 算法代码直接依赖 scheduler 内部实现。
2. 将主仓库接口差异收敛在 `adapters/vllm_hust/` 中。主仓库升级时，优先修改 adapter 与 manifest，不要让业务算法散落依赖 `vllm` 内部实现。
3. 在同一个 Python 环境中安装目标 vLLM-HUST 与刚构建的插件 wheel，确认 `vllm` 命令和插件都来自该环境。
4. 设置 manifest 和 Bundle allowlist，先调用 `get_configured_extension_startup().diagnostics()` 验证准入结果；再启动最小服务，检查 startup snapshot 和 typed victim selector 的 provenance 日志，并执行健康检查和一个能触发抢占的请求集。
5. 在 CI 中至少覆盖最低支持版本和当前最新支持版本。每个发布候选都要在相同的主仓库版本、模型、设备、请求集和并发条件下与 baseline 比较功能或性能。

主仓库升级、contract 变更、插件改动或设备后端变更时，都应重新执行上述流程，并更新 manifest 的 `host_api_range`、兼容性表和插件版本。提交 `5c994cdc` 本身没有提供 `vllm plugin list/inspect/validate` 或 `--extension` CLI；不应使用这些命令判断该版本是否支持 Bundle v1。

## 6. 版本号

BidKV 从 `src/bidkv/_version.py` 读取发行版本：

```python
__version__ = "0.1.0"
```

同一次发布中，Python 发行版本和 manifest 的 `bundle_version` 保持一致。PyPI 不允许覆盖同一版本下已经上传的文件，修改源码后需要增加版本号再重新构建。

## 7. 构建 wheel 和 sdist

在项目根目录记录本次发布对应的 commit，并检查工作树：

```bash
git status --short
git rev-parse HEAD
```

清理旧产物并构建：

```bash
rm -rf dist
uv build --no-sources --out-dir dist
```

PowerShell 使用：

```powershell
if (Test-Path -LiteralPath dist) {
    Remove-Item -LiteralPath dist -Recurse -Force
}
uv build --no-sources --out-dir dist
```

`--no-sources` 会忽略本地 `tool.uv.sources` 覆盖，更接近普通用户的构建环境。

以 `bidkv==0.1.0` 为例，`dist/` 中会生成：

```text
dist/
├── bidkv-0.1.0-py3-none-any.whl
└── bidkv-0.1.0.tar.gz
```

纯 Python 插件通常生成 `py3-none-any` wheel。包含编译扩展的项目需要为支持的 Python ABI 和平台分别构建 wheel。

## 8. 检查构建产物

列出 wheel 的内容：

```bash
python -m zipfile -l dist/bidkv-0.1.0-py3-none-any.whl
```

BidKV wheel 中至少应有：

```text
bidkv/__init__.py
bidkv/_version.py
bidkv/manifests/vllm-hust-extension-v1.json
bidkv-0.1.0.dist-info/METADATA
bidkv-0.1.0.dist-info/RECORD
```

如果包含 legacy 兼容 entry point，wheel 还应包含 `entry_points.txt`。可以这样检查：

```bash
python -c 'import zipfile; p="dist/bidkv-0.1.0-py3-none-any.whl"; z=zipfile.ZipFile(p); n=next(x for x in z.namelist() if x.endswith(".dist-info/entry_points.txt")); print(z.read(n).decode())'
```

输出中应包含：

```ini
[vllm.victim_selector]
bidkv = bidkv.adapters.vllm_hust.selector:BidkvVictimSelector
```

typed Bundle 路径只要求 wheel 包含 manifest，不要求 `entry_points.txt`。

然后把 wheel 安装到临时环境：

```bash
uv venv .release-smoke
uv pip install --python .release-smoke/bin/python \
  --no-deps dist/bidkv-0.1.0-py3-none-any.whl
.release-smoke/bin/python -c \
  'from importlib.metadata import version; print(version("bidkv"))'
```

PowerShell：

```powershell
uv venv .release-smoke
uv pip install --python .release-smoke\Scripts\python.exe `
  --no-deps dist\bidkv-0.1.0-py3-none-any.whl
.\.release-smoke\Scripts\python.exe -c `
  'from importlib.metadata import version; print(version("bidkv"))'
```

临时环境中装有目标版本的 vLLM-HUST 时，显式定位 manifest 并验证准入结果：

```bash
BIDKV_MANIFEST="$(.release-smoke/bin/python -c \
  'from importlib.resources import files; print(files("bidkv").joinpath("manifests", "vllm-hust-extension-v1.json"))')"
VLLM_EXTENSION_MANIFESTS="$BIDKV_MANIFEST" \
VLLM_EXTENSION_BUNDLES="org.vllm-hust.bidkv" \
  .release-smoke/bin/python -c \
  'from vllm.plugins.startup import get_configured_extension_startup; print(get_configured_extension_startup().diagnostics())'
```

预期 diagnostics 中的 `admitted_bundle_ids` 包含 `org.vllm-hust.bidkv`，`admitted_component_ids` 包含 `org.vllm-hust.bidkv/victim-selector`，`disabled_bundle_ids` 为空。

## 9. 发布到 PyPI

### 项目和权限

首次发布前，需要确定 PyPI 项目名并准备上传权限。BidKV 的 PyPI 项目名是 `bidkv`，项目归属 `intellistream` Organization。

团队发布可以使用组织或项目管理页面提供的 API Token。只发布一个项目时，使用仅授权该项目的 Token；多项目发布流水线才需要更大的授权范围。

Token 保存在 CI Secret 或密码库中，不写入仓库、`pyproject.toml`、构建脚本和日志。PyPI Token 以 `pypi-` 开头；上传用户名为 `__token__`。设置 `UV_PUBLISH_TOKEN` 后，`uv` 会处理用户名。

### 上传

在发布环境中从安全位置读取 Token：

```bash
export UV_PUBLISH_TOKEN='<从 CI Secret 或密码库读取>'
```

上传 BidKV 的 wheel 和 sdist：

```bash
uv publish \
  --check-url https://pypi.org/simple \
  dist/bidkv-0.1.0-py3-none-any.whl \
  dist/bidkv-0.1.0.tar.gz
```

发布自己的插件时替换两个文件名。正式发布建议显式列出文件，避免把 `dist/` 中残留的其他版本一起上传。

`--check-url` 用于检查索引中是否已有相同文件。如果一次发布只上传了部分文件，可以在本地产物没有变化的前提下重试；如果远端同版本文件与本地内容不同，应改用新版本号。

发布结束后清理当前 shell 中的 Token：

```bash
unset UV_PUBLISH_TOKEN
```

PowerShell：

```powershell
Remove-Item Env:UV_PUBLISH_TOKEN
```

### CI 示例

下面以 GitHub Actions 为例，Token 保存在 `PYPI_TOKEN` Secret 中：

```yaml
- name: Build distributions
  run: uv build --no-sources --out-dir dist

- name: Publish package
  env:
    UV_PUBLISH_TOKEN: ${{ secrets.PYPI_TOKEN }}
  run: >-
    uv publish
    --check-url https://pypi.org/simple
    dist/bidkv-0.1.0-py3-none-any.whl
    dist/bidkv-0.1.0.tar.gz
```

发布 job 一般只允许受保护的 tag 或 release 分支触发，并确保构建、测试和上传使用同一个 commit。版本号可以从项目元数据读取，避免长期写死在 YAML 中。

## 10. 从 PyPI 安装

用户按 PyPI 项目名安装插件。BidKV 的安装命令是：

```bash
python -m pip install "bidkv==0.1.0"
```

使用 uv：

```bash
uv pip install --python /path/to/vllm-env/bin/python "bidkv==0.1.0"
```

插件需要安装到运行 `vllm` 命令的同一个 Python 环境：

```bash
command -v python
command -v vllm
python -c 'import sys; print(sys.executable)'
python -m pip show bidkv
```

发布后还要从正式 PyPI 做一次无缓存安装：

```bash
uv venv .pypi-smoke
uv pip install --python .pypi-smoke/bin/python \
  --no-cache --refresh-package bidkv "bidkv==0.1.0"
.pypi-smoke/bin/python -c \
  'from importlib.metadata import version; assert version("bidkv") == "0.1.0"'
```

安装后先确认 manifest 可以从当前环境定位：

```bash
python -c \
  'from importlib.resources import files; print(files("bidkv").joinpath("manifests", "vllm-hust-extension-v1.json"))'
```

这只证明包和 manifest 已安装。Bundle 准入和实现加载需要按下一节配置并启动 vLLM-HUST。

## 11. 启用插件

安装 Python 包只是把代码和 manifest 放入当前环境。启动时还必须显式传入 manifest，可选地用 allowlist 准入 Bundle，并在有多个 scheduler-policy provider 时选择具体组件。

BidKV 的启动示例：

```bash
BIDKV_MANIFEST="$(python -c \
  'from importlib.resources import files; print(files("bidkv").joinpath("manifests", "vllm-hust-extension-v1.json"))')"
export VLLM_EXTENSION_MANIFESTS="$BIDKV_MANIFEST"
export VLLM_EXTENSION_BUNDLES="org.vllm-hust.bidkv"

vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --additional-config '{
    "victim_selector_component": "org.vllm-hust.bidkv/victim-selector",
    "enable_utility_victim_selection": true,
    "utility_strategy": "bidkv",
    "utility_kv_gate": 0.95
  }'
```

这里有三层配置：

- `VLLM_EXTENSION_MANIFESTS` 显式配置 manifest。Linux/macOS 的多路径分隔符是 `:`，Windows 是 `;`。
- `VLLM_EXTENSION_BUNDLES` 是逗号分隔的 Bundle ID allowlist；未设置时准入所有已显式配置的 manifest。在 `5c994cdc` 中不要把它设为空字符串：实际解析会产生空 Bundle ID，并以“allowlist 中存在未配置 ID”报错。需要停用时移除 manifest 配置，或使用 `victim_selector_plugin_disabled`。
- `victim_selector_component` 选择 Bundle 中的具体组件。只有一个 typed provider 时可省略，显式写出更利于审计和避免后续新增 provider 造成启动歧义。

其他插件根据自己的 contract 使用相应的运行配置，不能直接复制 `victim_selector_component`。启动日志中应能看到被准入的 Bundle 和组件：

`enable_utility_victim_selection`、`utility_strategy` 和 `utility_kv_gate` 是 BidKV adapter 从 `vllm_config.additional_config` 消费的业务配置，不是 Extension Bundle 宿主的通用字段。发布新版 BidKV 时应以该版 adapter 实际支持的键为准。

```text
admitted_bundles=('org.vllm-hust.bidkv',)
disabled_bundles=()
admitted_components=('org.vllm-hust.bidkv/victim-selector',)
Loaded typed victim selector component=org.vllm-hust.bidkv/victim-selector ... api_version=1
```

在启动服务前，也可以用同一环境做一次不导入 BidKV 实现的静态准入检查：

```bash
python -c \
  'from vllm.plugins.startup import get_configured_extension_startup; print(get_configured_extension_startup().diagnostics())'
```

上面的 Python 静态检查会解析和准入 manifest，但不会导入 `implementation_ref`。只有 scheduler 初始化并出现 `Loaded typed victim selector` 日志才证明实现已物化。

服务启动后可以做健康检查：

```bash
curl --fail http://127.0.0.1:8000/health
```

插件的功能或性能还需要在相同模型、设备、请求集和并发条件下与 baseline 对比。

## 12. 升级、停用和卸载

停止使用插件的 vLLM 实例，再升级包：

```bash
python -m pip install --no-cache-dir --upgrade "bidkv==<新版本>"
python -c \
  'from importlib.resources import files; print(files("bidkv").joinpath("manifests", "vllm-hust-extension-v1.json"))'
```

检查新 manifest 中的 `bundle_version` 与 Python 包版本一致，并重新运行静态准入检查。运行中的 Python 进程不会自动重新加载刚安装的代码，而且 startup resolution 在进程内有缓存，所以升级后必须启动新进程。需要回退时安装之前验证过的版本：

```bash
python -m pip install --no-cache-dir --force-reinstall "bidkv==0.1.0"
```

停用 BidKV 时，从下一次启动环境中移除 BidKV manifest，或从 allowlist 中移除它，并删除组件选择：

```text
VLLM_EXTENSION_MANIFESTS 中的 BidKV manifest 路径
VLLM_EXTENSION_BUNDLES 中的 org.vllm-hust.bidkv
victim_selector_component=org.vllm-hust.bidkv/victim-selector
```

如果需要在保留部署配置的情况下强制使用 upstream 选择行为，可在 `--additional-config` 中设置 `"victim_selector_plugin_disabled": true`。该开关优先级最高。

卸载前先停止相关 vLLM 实例并清理启动配置：

```bash
python -m pip uninstall bidkv
python -c 'import importlib.util; assert importlib.util.find_spec("bidkv") is None'
```

卸载后必须同时清理 `VLLM_EXTENSION_MANIFESTS`、`VLLM_EXTENSION_BUNDLES` 和 `victim_selector_component`，否则下次启动会因 manifest 不存在或组件未准入而 fail closed。如果模块卸载后仍可导入，通常是插件与 `vllm` 不在同一个 Python 环境，或 editable 安装仍有元数据残留。先用 `python -m pip show bidkv` 定位安装环境，不要直接删除单个 `.py` 文件。

## 13. 常见问题

### wheel 中没有 manifest

检查 `pyproject.toml` 中的 package data，清理 `dist/` 后重新构建：

```toml
[tool.setuptools.package-data]
bidkv = ["manifests/*.json"]
```

### startup diagnostics 找不到 Bundle

常见原因有：

- 插件与 `vllm` 安装在不同的 Python 环境；
- wheel 缺少 manifest；
- `VLLM_EXTENSION_MANIFESTS` 没有包含该 manifest 的正确路径；
- `VLLM_EXTENSION_BUNDLES` allowlist 没有包含 `org.vllm-hust.bidkv`。

### 静态准入报 host API 不兼容

插件声明的 `host_api_range` 与当前 vLLM-HUST 不匹配。应升级插件、切换到兼容的 vLLM-HUST，或发布声明正确兼容范围的新版本。

### PyPI 返回 `403`

检查 Token 是否完整、是否已撤销、是否有目标项目的上传权限，以及环境变量中是否带有换行或空格。PyPI 与 TestPyPI 使用不同的 Token。

### PyPI 提示文件已经存在

同一版本的文件不能覆盖。确认远端内容，增加版本号后重新构建和上传。

### 安装后插件行为没有变化

确认启动配置中既有 manifest、Bundle 准入和组件选择，也有 BidKV 自身的业务开关。以 BidKV 为例：

```text
VLLM_EXTENSION_MANIFESTS=<BidKV manifest 绝对路径>
VLLM_EXTENSION_BUNDLES=org.vllm-hust.bidkv
victim_selector_component=org.vllm-hust.bidkv/victim-selector
enable_utility_victim_selection=true
```

`pip show bidkv` 只能说明包已安装，不能说明运行中的服务已经启用 BidKV。检查日志中是否出现 `Loaded typed victim selector component=org.vllm-hust.bidkv/victim-selector`。

## 14. 检查表

### 插件配置

- [ ] PyPI 项目名和 Python 模块名已经确定。
- [ ] Bundle ID 和 Component ID 唯一且稳定。
- [ ] contract、execution plane、isolation 和 permissions 与实现一致。
- [ ] manifest 已加入 package data。
- [ ] `implementation_ref` 使用可导入的 `module:attribute` 形式。
- [ ] 实现声明 API version `1`，并完整实现 `VictimSelector` protocol。
- [ ] 如果保留 legacy 兼容，`vllm.victim_selector` entry-point name 为 `bidkv`。
- [ ] 发行版本与 `bundle_version` 一致。
- [ ] 已记录并验证目标 vLLM-HUST 版本或 commit、host API 范围、contract 和运行环境。
- [ ] 主仓库的最低支持版本与最新支持版本均已完成插件集成测试。

### 构建与发布

- [ ] 测试通过，发布 commit 已记录。
- [ ] `dist/` 中只有本次发布的 wheel 和 sdist。
- [ ] wheel 包含 manifest 和发行元数据；启用 legacy 兼容时还包含对应 entry point。
- [ ] wheel 的隔离安装测试通过。
- [ ] API Token 的权限范围符合发布需要。
- [ ] PyPI 上传成功，文件名和哈希已记录。

### 安装与使用

- [ ] 已从正式 PyPI 无缓存安装。
- [ ] 安装版本与发布版本一致。
- [ ] `VLLM_EXTENSION_MANIFESTS` 指向已安装 wheel 中的 manifest。
- [ ] startup diagnostics 显示 Bundle 和 component 已准入。
- [ ] 多 provider 时已用完整 qualified ID 选择组件。
- [ ] 启动日志显示 typed victim selector 已物化。
- [ ] 服务完成健康检查。
- [ ] 功能或性能效果经过单独验收。

## 参考资料

- [BidKV typed scheduler-policy 适配提交 `5c994cdc`](https://github.com/vLLM-HUST/vllm-hust/commit/5c994cdc029dfebe318ca745a39920473033038b)
- [vLLM-HUST 核心运行时仓库](https://github.com/vLLM-HUST/vllm-hust)
- [PyPI Organization Accounts](https://docs.pypi.org/organization-accounts/)
- [PyPI API Token 帮助](https://pypi.org/help/#apitoken)
- [uv：构建与发布 Python 包](https://docs.astral.sh/uv/guides/package/)
- [uv 发布相关环境变量](https://docs.astral.sh/uv/configuration/environment/)
