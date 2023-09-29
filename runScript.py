import os
import subprocess
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

weights_path = "./runs/train/exp4/weights/best.pt"
image_size = 640
confidence_threshold = 0.25
source_folder = './buffer'

def test_yolov5():
    command = f'python detect.py --weight {weights_path} --img {image_size} --conf {confidence_threshold} --source {source_folder} --save-csv'
    subprocess.run(command, shell=True)
    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {str(e)}")

class BufferHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.running = True  # ตัวแปรสำหรับติดตามสถานะการทำงาน

    def on_created(self, event):
        if event.is_directory:
            return
        elif event.event_type == 'created':
            print(f"Detected new file: {event.src_path}")
            test_yolov5()

    def on_modified(self, event):
        if event.is_directory:
            return
        elif event.event_type == 'modified' and not os.listdir(source_folder):
            # หยุดการทำงานเมื่อไม่มีไฟล์ในโฟลเดอร์ Buffer
            print("No files in the Buffer folder. Stopping.")
            self.running = False

observer = Observer()
buffer_hander = BufferHandler()
observer.schedule(BufferHandler(), path=source_folder, recursive=False)
observer.start()

try:
    print("Auto YOLOv5 testing is running. Press Ctrl+C to stop.")
    while buffer_hander.running:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
    observer.join()