import cv2
import numpy as np
import math

# 视频输出设置
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# 人脸检测器
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# 光流法参数（优化跟踪性能）
lk_params = dict(winSize=(35, 35),  # 增大窗口尺寸
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

# 运动检测参数（优化后的灵敏度参数）
NOD_THRESHOLD = 20        # 降低垂直阈值
SHAKE_THRESHOLD = 10      # 降低水平阈值
DIRECTION_CHANGES = 2     # 减少方向变化要求
DOMINANCE_RATIO = 1.5     # 降低主导方向要求
TRACKING_FRAMES = 15      # 缩短分析窗口
MIN_MOVE = 2              # 降低有效移动阈值
DISPLAY_DURATION = 45     # 显示持续时间增加

def get_coords(p):
    try: return int(p[0][0][0]), int(p[0][0][1])
    except: return int(p[0][0]), int(p[0][1])

class GestureDetector:
    def __init__(self):
        self.track_points = []
        self.direction_history = []
        self.gesture_status = {"nod": 0, "shake": 0}
        self.face_center = None
        self.lost_counter = 0
        self.debug_info = {}  # 新增调试信息存储

    def update_tracking(self, new_point):
        self.track_points.append(new_point)
        if len(self.track_points) > TRACKING_FRAMES:
            self.track_points.pop(0)

        if len(self.track_points) >= 2:
            prev = self.track_points[-2]
            curr = self.track_points[-1]
            dx = curr[0][0][0] - prev[0][0][0]
            
            # 优化方向检测逻辑
            if abs(dx) > MIN_MOVE:
                direction = 1 if dx > 0 else -1
                self.direction_history.append(direction)
                # 保持方向历史在合理范围内
                if len(self.direction_history) > 20:
                    self.direction_history = self.direction_history[-20:]

    def analyze_motion(self):
        if len(self.track_points) < 2:
            return None

        # 计算窗口内总位移（优化算法）
        start_point = self.track_points[0]
        end_point = self.track_points[-1]
        total_x = end_point[0][0][0] - start_point[0][0][0]
        total_y = end_point[0][0][1] - start_point[0][0][1]
        
        # 存储调试信息
        self.debug_info = {
            'total_x': total_x,
            'total_y': total_y,
            'abs_x': abs(total_x),
            'abs_y': abs(total_y)
        }

        # 优化运动主导性判断（添加最小运动量检查）
        min_motion = max(NOD_THRESHOLD, SHAKE_THRESHOLD) * 0.6
        if abs(total_x) + abs(total_y) < min_motion:
            return None
            
        # 动态比例计算（防止除零错误）
        sum_xy = abs(total_x) + abs(total_y)
        x_ratio = abs(total_x) / sum_xy if sum_xy != 0 else 0
        y_ratio = abs(total_y) / sum_xy if sum_xy != 0 else 0
        
        is_x_dominant = x_ratio > (y_ratio * DOMINANCE_RATIO)
        is_y_dominant = y_ratio > (x_ratio * DOMINANCE_RATIO)

        # 点头检测（优化触发条件）
        if is_y_dominant and abs(total_y) > NOD_THRESHOLD:
            return "nod"
            
        # 摇头检测（优化方向变化检测）
        if is_x_dominant and abs(total_x) > SHAKE_THRESHOLD:
            # 计算有效方向变化
            dir_changes = sum(1 for i in range(1, len(self.direction_history)) 
                        if self.direction_history[i] != self.direction_history[i-1])
            if dir_changes >= DIRECTION_CHANGES:
                return "shake"
            
        return None

    def reset_counters(self, detected_gesture):
        if detected_gesture == "nod":
            self.gesture_status["nod"] = DISPLAY_DURATION
            self.direction_history = []
        elif detected_gesture == "shake":
            self.gesture_status["shake"] = DISPLAY_DURATION
            self.track_points = self.track_points[-5:]

def main():
    cap = cv2.VideoCapture(0)
    detector = GestureDetector()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 优化人脸检测（提高检测频率）
        if detector.face_center is None or detector.lost_counter > 5:
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            if len(faces) > 0:
                x, y, w, h = faces[0]
                # 优化中心点计算（使用下巴位置）
                detector.face_center = (x + w//2, y + h//2 + h//5)
                detector.track_points = [np.array([[detector.face_center]], np.float32)]
                detector.lost_counter = 0
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            else:
                detector.lost_counter += 1
                detector.face_center = None
                # 显示提示信息
                cv2.putText(frame, "Searching Face...", (50, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow('Gesture Detection', frame)
                continue

        # 光流跟踪（添加错误处理）
        if detector.track_points:
            try:
                prev_gray = gray.copy() if len(detector.track_points) == 1 else prev_gray
                new_points, st, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, detector.track_points[-1], None, **lk_params)
                
                if new_points is not None:
                    detector.update_tracking(new_points)
                    prev_gray = gray.copy()
                    
                    # 运动分析
                    gesture = detector.analyze_motion()
                    if gesture:
                        detector.reset_counters(gesture)
            except Exception as e:
                print(f"Tracking error: {str(e)}")
                detector.track_points = []
        
        # 显示结果（优化可视化）
        if detector.gesture_status["nod"] > 0:
            cv2.putText(frame, "NODDING", (50, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            detector.gesture_status["nod"] -= 1
            
        if detector.gesture_status["shake"] > 0:
            cv2.putText(frame, "SHAKING", (50, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 3)
            detector.gesture_status["shake"] -= 1
        
        # 增强调试信息显示
        cv2.putText(frame, f"X: {detector.debug_info.get('total_x', 0):.0f}", (20, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 1)
        cv2.putText(frame, f"Y: {detector.debug_info.get('total_y', 0):.0f}", (20, 210),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 1)
        cv2.putText(frame, f"Dir: {len(detector.direction_history)}", (20, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 1)
        cv2.putText(frame, f"XY Ratio: {detector.debug_info.get('abs_x',0)/(detector.debug_info.get('abs_y',1)+1e-5):.1f}", 
                   (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 1)
        
        # 绘制运动轨迹
        for point in detector.track_points:
            x, y = get_coords(point)
            cv2.circle(frame, (x, y), 3, (0,255,0), -1)
        
        cv2.imshow('Gesture Detection', frame)
        out.write(frame)
        
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()