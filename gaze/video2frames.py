import cv2
import os
from PIL import Image

# 输入视频路径
video_path = "video/input/video/nod.mp4"
# 输出图像文件夹
output_dir = "video/input/frames"

# 创建输出文件夹
os.makedirs(output_dir, exist_ok=True)

# 打开视频
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

# 获取视频帧数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {total_frames}")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    # 将BGR图像转换为RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 使用PIL库打开图像进行调整大小
    image = Image.fromarray(frame_rgb)
    image_resized = image.resize((224, 224))
    
    # 保存图像，格式为 frame_0001.jpg, frame_0002.jpg, ...
    output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
    image_resized.save(output_path)
    print(f"Saved frame {frame_count}/{total_frames} to {output_path}")

# 释放资源
cap.release()
cv2.destroyAllWindows()
print("Video to frames conversion and resizing completed.")
