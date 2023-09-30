import os
import io
import subprocess
import time
import mysql.connector
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from googleapiclient.discovery import build
from google.oauth2 import service_account

mysql_config = {
    'user': 'your_username',
    'password': 'your_password',
    'host': 'localhost',
    'database': 'your_database',
}

SCOPES = ['https://www.googleapis.com/auth/drive.file']
SERVICE_ACCOUNT_FILE = 'your-service-account-json-file.json'

conn = mysql.connector.connect(**mysql_config)
cursor = conn.cursor()

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

weights_path = "./runs/train/exp4/weights/best.pt"
image_size = 640
confidence_threshold = 0.25
source_folder = './buffer'
temp_folder = './runs/detect'

def delete_file(src):
    try:
        for root, dirs, files in os.walk(src, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                os.rmdir(dir_path)
        print(f"Delect file in {src} complete.")
    except Exception as e:
        print(f"Error deleting file {file_path}: {str(e)}")

def test_yolov5():
    print(f"Detected new file in the Buffer folder. detect.py is Running.")
    command = f'python detect.py --weight {weights_path} --img {image_size} --conf {confidence_threshold} --source {source_folder} --save-csv'
    subprocess.run(command, shell=True)
    delete_file(source_folder)

class BufferHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        print("Auto YOLOv5 testing is running. Press Ctrl+C to stop.")
        if os.listdir(source_folder):
            test_yolov5()

    def on_created(self, event):
        if event.event_type == 'created' and os.listdir(source_folder):
            test_yolov5()

observer = Observer()
observer.schedule(BufferHandler(), path=source_folder, recursive=False)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    delete_file(temp_folder)
    observer.stop()
    observer.join()
