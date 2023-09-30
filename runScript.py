import os
import subprocess
import time
import datetime
import csv
import mysql.connector
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

mysql_config = {
    'user': 'root',
    'host': 'localhost',
    'database': 'warehouse_db',
}

conn = mysql.connector.connect(**mysql_config)
cursor = conn.cursor()

weights_path = "./runs/train/exp4/weights/best.pt"
image_size = 640
confidence_threshold = 0.25
source_folder = './buffer'
temp_folder = './runs/detect/exp'

def delete_file(source):
    try:
        for root, dirs, files in os.walk(source, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                os.rmdir(dir_path)
        print(f"Delect file in {source} complete.")
    except Exception as e:
        print(f"Error deleting file {file_path}: {str(e)}")

def process_prediction_csv():
    try:
        with open(f'{temp_folder}/predictions.csv', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if len(row) >= 3:
                    filename = row[0]
                    classification = row[1]
                    confidence = float(row[2])
                    timestamp = datetime.datetime.now()
                    original_image_path = os.path.join(source_folder, filename)
                    result_image_path = os.path.join(temp_folder, filename)

                    insert_data_into_mysql(filename, classification, confidence, timestamp, original_image_path, result_image_path)
            
            delete_file(f"../{temp_folder}")

    except Exception as e:
        print(f"Error processing predictions.csv: {str(e)}")

def insert_data_into_mysql(filename, classification, confidence, timestamp, original_image_path, result_image_path):
    try:
        sql = "INSERT INTO cement_stock_list (file_name, classification, confidence, timestamp, original_image_path, result_image_path) VALUES (%s, %s, %s, %s, %s, %s)"
        values = (filename, classification, confidence, timestamp, original_image_path, result_image_path)
        cursor.execute(sql, values)
        conn.commit()
        print(f"Data inserted into MySQL: {filename}, {classification}, {confidence}, {timestamp}")
    except Exception as e:
        print(f"Error inserting data into MySQL: {str(e)}")

def test_yolov5():
    print(f"Detected new file in the Buffer folder. detect.py is Running.")
    command = f'python detect.py --weight {weights_path} --img {image_size} --conf {confidence_threshold} --source {source_folder} --save-csv'
    subprocess.run(command, shell=True)
    process_prediction_csv()
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
    observer.stop()
    observer.join()
