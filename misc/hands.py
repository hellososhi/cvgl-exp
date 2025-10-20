import cv2
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import time


def main():
    # カメラを開く（デフォルトカメラ=0）
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: Camera not opened")
        return

    # MediaPipe Hands の初期化
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        prev_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame from camera")
                break

            # BGR -> RGB に変換
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # ランドマーク検出
            results = hands.process(image)

            # 描画のために再び書き込み可能にして BGR に戻す
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # ランドマークを描画
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            # FPS 表示
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0.0
            prev_time = curr_time
            cv2.putText(
                image,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("MediaPipe Hands", image)

            # 'q' キーで終了
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
