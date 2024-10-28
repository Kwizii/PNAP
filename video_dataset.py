import cv2
import os


def extract_frames(video_path, output_folder, frame_interval=10):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    v_name = os.path.splitext(os.path.basename(video_path))[0]
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    count = 0

    # 提取帧并保存为图像文件
    while success:
        if count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"{v_name}_{count}.jpg")
            cv2.imwrite(frame_path, frame)  # 保存帧为JPEG图像
        success, frame = cap.read()
        count += 1

    cap.release()


def process_videos(directory):
    # 遍历目录中的视频文件并处理每个视频
    for filename in os.listdir(directory):
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            video_path = os.path.join(directory, filename)
            extract_frames(video_path, directory)
            print(f"视频 {filename} 处理完成.")


if __name__ == "__main__":
    videos_directory = "../datasets/sandtable/video"
    process_videos(videos_directory)
