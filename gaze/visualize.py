# visualize_all_gaze.py
import os, cv2, numpy as np
from glob import glob

# === config =========================
frames_dir = r"video\input\frames"                  # 原始帧
pred_dir   = r"weight\gaze360\frames_infer"         # 存放 *_gaze_pred*.txt
out_dir    = r"video\output\vis"                    # 保存可视化
# ===================================================


def gazeto3d(g):
    """[yaw,pitch] -> 3D gaze向量"""
    v = np.zeros(3)
    v[0] = -np.cos(g[1]) * np.sin(g[0])
    v[1] = -np.sin(g[1])
    v[2] = -np.cos(g[1]) * np.cos(g[0])
    return v

def draw_gaze(img, gaze, scale=150,
              color=(0,0,255), thick=4, tip=0.25):
    if len(gaze)==2:
        gaze = gazeto3d(gaze)
    H,W = img.shape[:2]
    start = (W//2, H//2)
    end = (int(start[0]-gaze[0]*scale),
           int(start[1]+gaze[1]*scale))
    cv2.arrowedLine(img, start, end, color, thick,
                    tipLength=tip, line_type=cv2.LINE_AA)
    return img


def load_preds(file_path):
    """读取单个 *_gaze_pred.txt -> dict{name: np.array([yaw,pitch])}"""
    d = {}
    with open(file_path,'r',encoding='utf-8') as f:
        for ln in f.readlines()[1:]:
            name, gz = ln.strip().split()
            d[name] = np.array(list(map(float, gz.split(','))))
    return d


def main():
    os.makedirs(out_dir, exist_ok=True)
    txt_list = glob(os.path.join(pred_dir, "*gaze_pred*.txt"))
    if not txt_list:
        print("⛔ 未找到任何 *_gaze_pred*.txt")
        return
    print(f"[i] 共发现 {len(txt_list)} 个预测文件")

    missing = 0
    total   = 0
    for txt in txt_list:
        preds = load_preds(txt)
        print(f"  ↳ 处理 {os.path.basename(txt)} : {len(preds)} 条")
        for rel, gaze in preds.items():
            total += 1
            src = os.path.join(frames_dir, rel)
            if not os.path.isfile(src):
                print(f"[!] 缺少图像 {src}")
                missing += 1
                continue
            img = cv2.imread(src)
            if img is None:
                print(f"[!] 读取失败 {src}")
                missing += 1
                continue

            vis = draw_gaze(img, gaze)
            dst = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            cv2.imwrite(dst, vis)

    print(f"\n[✓] 完成！共处理 {total} 张，缺失/失败 {missing} 张")
    print(f"    可视化结果已保存到 {out_dir}")


if __name__ == "__main__":
    main()
