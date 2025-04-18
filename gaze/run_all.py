# run_all.py ── 一键流水线（按顺序运行）
import subprocess, sys, os

# === 配置路径：4 个脚本的位置 ===
video2frames   = r"video2frames.py"          # 根目录下的 video2frames.py
test_no_label  = r"tester/test_no_label.py"  # tester 目录下的 test_no_label.py
visualize      = r"visualize.py"              # 根目录下的 visualize.py
frames2video   = r"frames2video.py"           # 根目录下的 frames2video.py

# === 执行命令的通用方法 ===
def run(cmd):
    print(f"\n=== 执行: {' '.join(cmd)} ===")
    subprocess.check_call(cmd)

if __name__ == "__main__":
    # 执行各步骤的命令
    commands = [
        [sys.executable, video2frames],   # Step‑1  video → frames
        [sys.executable, test_no_label],  # Step‑2  推理
        [sys.executable, visualize],      # Step‑3  可视化
        [sys.executable, frames2video]    # Step‑4  frames → video
    ]
    
    # 按顺序执行
    for cmd in commands:
        run(cmd)
    
    print("\n🎉  流水线任务完成！")
