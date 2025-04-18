import cv2
import os

# 设置图片文件夹路径
image_folder = 'video/output/vis'
# 输出视频的路径
output_video_path = 'output_video.mp4'

# 设定每秒帧数（FPS）
fps = 30

# 获取文件夹中所有图片文件，并排序
images = [img for img in os.listdir(image_folder) if img.endswith(('.jpg', '.jpeg', '.png'))]
images.sort()  # 确保顺序正确，比如 frame_0001.jpg, frame_0002.jpg, ...

# 读取第一张图片确定尺寸
first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)
height, width, layers = frame.shape
size = (width, height)

# 定义视频编码器（mp4v适用于.mp4文件）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video_path, fourcc, fps, size)

# 将每张图片写入视频
for image_name in images:
    image_path = os.path.join(image_folder, image_name)
    frame = cv2.imread(image_path)
    video.write(frame)

# 释放资源
video.release()
cv2.destroyAllWindows()

print("Video creation completed!")
