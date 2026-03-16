import cv2
import mediapipe as mp
import pyautogui
import math
# 變數   # 3/15 可以執行：食指+拇指=左鍵 中指+拇指=右鍵 食指+中指向上/下=滾輪 握拳=滑鼠不動
speed_factor =  1 # 滑鼠倍數
left_click_cooldown = 0
right_click_cooldown = 0
scroll_prev_y = None
scroll_mode = False

# Tasks API 的正確模組
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 這裡把模型路徑換成你下載的 .task 路徑
model_path = "hand_landmarker.task"

# 設定偵測參數
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
)

# 建立偵測器
landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()
# 初始化上一幀手指座標
prev_hand_x = None
prev_hand_y = None
was_fist = False

smoothening = 1
frame_timestamp_ms = 0

# --------- 先定義判斷握拳函式 ---------
def is_fist(hand):
    # landmark 0 是手腕，8-20 是指尖
    wrist = hand[0]
    finger_tips = [hand[8], hand[12], hand[16], hand[20]]  # 食指、中指、無名指、小指
    distances = [math.hypot((tip.x - wrist.x), (tip.y - wrist.y)) for tip in finger_tips]
    avg_dist = sum(distances) / len(distances)
    return avg_dist < 0.2  # 閾值可調

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # MediaPipe Image 物件
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 偵測並取得 landmark
    results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    frame_timestamp_ms += 1

    if results.hand_landmarks:
        hand = results.hand_landmarks[0]

        # ---- 在這裡加 landmark 顯示 ----
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20)
        ]

        # 畫 landmark 點
        for lm in hand:
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # 畫骨架線
        for start, end in connections:
            x1 = int(hand[start].x * frame.shape[1])
            y1 = int(hand[start].y * frame.shape[0])
            x2 = int(hand[end].x * frame.shape[1])
            y2 = int(hand[end].y * frame.shape[0])
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # ---- landmark 顯示結束 ----

        # 取得食指和中指座標
        middle = hand[12]
        index = hand[8]
        thumb = hand[4]
        wrist = hand[0]

        # 判斷是否握拳
        fist = is_fist(hand)

        if fist:
            # 握拳 → 滑鼠停止
            was_fist = True

        else:
            hand_x = hand[0].x
            hand_y = hand[0].y

            # 如果是「剛從握拳變張開」
            if was_fist:
                prev_hand_x = hand_x
                prev_hand_y = hand_y
                was_fist = False
                continue

            # 如果是第一次偵測
            if prev_hand_x is None:
                prev_hand_x = hand_x
                prev_hand_y = hand_y
                continue

            # 計算移動
            dx = (hand_x - prev_hand_x) * screen_width * speed_factor / smoothening
            dy = (hand_y - prev_hand_y) * screen_height * speed_factor / smoothening

            pyautogui.moveRel(dx, dy)

            prev_hand_x = hand_x
            prev_hand_y = hand_y

        # 點擊判斷
        # 左鍵 (食指 + 拇指)
        dist_left = math.hypot(
            (thumb.x - index.x) * frame.shape[1],
            (thumb.y - index.y) * frame.shape[0]
        )

        if dist_left < 20 and left_click_cooldown == 0:
            pyautogui.click()
            left_click_cooldown = 5

        # 右鍵 (中指 + 拇指)
        dist_right = math.hypot(
            (thumb.x - middle.x) * frame.shape[1],
            (thumb.y - middle.y) * frame.shape[0]
        )

        if dist_right < 20 and right_click_cooldown == 0:
            pyautogui.rightClick()
            right_click_cooldown = 5
        
# ---------------- Scroll Control ----------------
        dist_scroll = math.hypot(
            (index.x - middle.x) * frame.shape[1],
            (index.y - middle.y) * frame.shape[0]
        )

        if dist_scroll < 20:

            # 兩指中心點
            scroll_y = (index.y + middle.y) / 2

            if not scroll_mode:

                scroll_mode = True

                # 進入 scroll mode 時停止滑鼠控制
                prev_hand_x = None
                prev_hand_y = None

            else:

                # 中立區域
                neutral_top = 0.45
                neutral_bottom = 0.55

                if scroll_y < neutral_top:
                    # 持續往上滾
                    pyautogui.scroll(200)

                elif scroll_y > neutral_bottom:
                    # 持續往下滾
                    pyautogui.scroll(-200)
        else:
            scroll_mode = False

    else:
        # 手消失 → 重置上一幀手指位置
        prev_hand_x = None
        prev_hand_y = None
    
    if left_click_cooldown > 0:
        left_click_cooldown -= 1

    if right_click_cooldown > 0:
        right_click_cooldown -= 1

    cv2.imshow("Hand Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()