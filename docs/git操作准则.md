# Git 操作准则（个人开发者版）

本项目由个人独立开发，Git 主要用于版本记录、回滚和备份。以下准则旨在保持提交历史的清晰、一致，方便日后追溯。

## 1. 分支策略

- **`main`**：主分支，始终处于可交付的稳定状态。所有开发工作不应直接在 `main` 上操作。
- **`dev`**（可选）：日常开发分支。如果项目简单，可以跳过，直接用 `feature/xxx` 分支并合并到 `main`。
- **`feature/xxx`**：功能分支，用于开发新功能、修复 bug、试验性修改。完成后合并回 `main`（或 `dev`）并删除该分支。

> 个人项目建议：直接从 `main` 切出 `feature/xxx`，开发完后合并回 `main`，避免分支过杂。

## 2. Commit 规范

采用 **Conventional Commits** 的简化形式，格式如下：

```
<类型>(<可选范围>): <简短描述>

[可选的详细说明]
```

### 2.1 类型（Type）

| 类型 | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | 修复 bug |
| `docs` | 仅文档更新（README、注释等） |
| `style` | 代码格式调整（空格、缩进、分号等），不影响逻辑 |
| `refactor` | 重构（既不是新功能也不是修 bug） |
| `perf` | 性能优化 |
| `test` | 添加或修改测试代码 |
| `chore` | 构建过程、辅助工具、配置文件等变更 |
| `revert` | 回退之前的提交 |

### 2.2 范围（Scope，可选）

用单词描述影响的模块，例如：`io`、`solver`、`postprocess`、`config`。  
如果没有明显范围，可以省略。

### 2.3 描述（Subject）

- 使用中文或英文均可，但要统一（建议中文，便于快速理解）
- 不超过 72 个字符
- 使用祈使句，不加句号
- 清晰说明“做了什么”，而非“怎么做的”

### 2.4 示例

```
feat(pipe): 添加各向同性管的压力计算函数

fix: 修复读取边界条件时数组越界错误

docs: 更新快速入门示例

refactor(solver): 提取公共线性代数运算到单独模块

chore: 添加 .gitignore 忽略 __pycache__/
```

## 3. 提交频率

- **每次完成一个可独立描述的小步骤**（例如：写完一个函数、修复一个错误、更新一段文档）即提交。
- 避免堆积大量修改后一次性提交，以免难以回滚或定位问题。
- 提交前确保代码至少能运行（不要求全面测试，但不应有明显语法错误）。

## 4. 标签（Tag）与版本管理

个人项目建议使用 **轻量标签**（lightweight tag）标记重要里程碑：

```bash
git tag v0.1.0
git push --tags
```

版本号采用 `v主版本.次版本.补丁`：

- 主版本：不兼容的重大重构
- 次版本：新增功能，兼容旧代码
- 补丁：bug 修复、文档改进等

## 5. 忽略文件（.gitignore）

Python 项目的常见忽略项示例：

```
# 字节码
__pycache__/
*.py[cod]

# 虚拟环境
venv/
.venv/
env/

# IDE 配置
.vscode/
.idea/

# 数据与临时文件
data/raw/*
!data/raw/.gitkeep
*.log
*.tmp

# 生成的文档
docs/_build/
```

## 6. 常用别名（可选）

为了提升效率，可以配置 Git 别名：

```bash
git config --global alias.ci "commit -m"
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.st status
git config --global alias.lg "log --oneline --graph --all"
```

## 7. 提交前快速自检清单

- [ ] 本次提交是否只包含一个逻辑单元？（不是两个不相关的修改混在一起）
- [ ] commit message 是否清晰描述了“做了什么”？
- [ ] 代码是否至少能通过语法检查？（没有未闭合的括号、明显缩进错误等）
- [ ] 是否误提交了敏感信息（密码、API key）或大文件？

---

> 以上准则可根据项目实际需求调整，关键是**保持一致**。个人开发时不必拘泥于形式，但好的习惯会让三个月后的自己感谢现在的你。