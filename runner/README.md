# 冲刺期间的自动化（GitHub 版）

数据流：**云端 Claude → Mac 上的 clone（`~/PycharmProjects/mic-pitch-class-steering`）→ `runner/mac_git_sync.sh` 自动 commit + push 到分支 `icassp` → GPU 机器 `runner/runner.sh` pull 并执行 `jobs/queue/*.sh` → 结果 push 回 `icassp` → 云端 pull 分析**。

## Mac（一次，之后一直挂着）

```bash
bash ~/PycharmProjects/mic-pitch-class-steering/runner/mac_git_sync.sh
```
它会自动建 `icassp` 分支并推上去，之后每 20 秒 add/commit/pull/push 一次。别在这个 clone 里手动改文件（会被一起提交，也没关系，但别和我改同一个文件）。

## GPU 机器（一次，等 GitHub 上出现 `icassp` 分支后）

```bash
cd ~ && git clone https://github.com/StoneYeY/mic-pitch-class-steering.git w2s && cd w2s && git checkout icassp
tmux new -d -s w2s 'bash runner/runner.sh'      # 看： tmux attach -t w2s   （Ctrl-b d 退出）
```
前提：这台机器上 `git push` 能用（有 credential / `gh auth login`）。conda 环境名默认 `mic`，不对就改 `runner/env.sh`。

## 之后

- `jobs/queue/NNN_name.sh` → runner 执行；`jobs/done/NNN_name.log|.exit|.progress`；小结果在 `results/NNN_name/`；大文件在 GPU 的 `runs/`（git-ignored）。
- `runner/status.json`：当前状态 + GPU 占用。
- 取消：提交一个空文件 `jobs/queue/<name>.cancel`。停 runner：GPU 上 `touch runner/STOP`。
