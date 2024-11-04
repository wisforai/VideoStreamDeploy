import json
from modules.deploy_handle import DeployHandle
from track.byte_tracker import BYTETracker
import subprocess
import time
import cv2
import base64
import os
import sys
import numpy as np
import socketio

sio = socketio.Client()


class Args:
    def __init__(self):
        self.mot20 = False
        self.track_thresh = 0.5
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.frame_rate = 30


args = Args()
tracker = BYTETracker(args)


def deploy(src_url, dst_url, uuid, person_deploy_handle):
    cap = cv2.VideoCapture(src_url)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    command = ['ffmpeg',
               '-y',
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-pix_fmt', 'bgr24',
               '-s', "{}x{}".format(width, height),
               '-r', str(fps),
               '-i', '-',
               '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p',
               '-f', 'flv',
               dst_url
               ]

    pipe = subprocess.Popen(command, stdin=subprocess.PIPE)

    max_reconnect_attempts = 5
    reconnect_attempts = 0
    previous_id = []
    current_id = []

    while cap.isOpened():
        time0 = time.time()
        ret, frame = cap.read()

        if not ret:
            print("Video stream disconnected. Attempting to reconnect to {}".format(src_url))
            reconnect_attempts += 1
            if reconnect_attempts > max_reconnect_attempts:
                print("Maximum reconnect attempts reached. Exiting.")
                break
            cap.release()
            cap = cv2.VideoCapture(src_url)
            continue
        else:
            reconnect_attempts = 0

        origin_frame = frame.copy()

        # 缩放图片
        diff = abs(height - width)
        if height > width:
            pad_img = cv2.copyMakeBorder(frame, 0, 0, 0, diff, cv2.BORDER_CONSTANT, 0)
        else:
            pad_img = cv2.copyMakeBorder(frame, 0, diff, 0, 0, cv2.BORDER_CONSTANT, 0)
        img_rs = cv2.resize(pad_img, (640, 640))

        time1 = time.time()
        result = person_deploy_handle.deploy(img_rs)
        time2 = time.time()
        print("Deploy time:", time2 - time1)
        msg = result['msg']
        if msg == '模型推理失败！':
            pipe.stdin.write(frame.tobytes())
            time3 = time.time()
            print("All time:", time3 - time0)
            continue

        data = result['data']

        if data[0] is None:
            pipe.stdin.write(frame.tobytes())
            time3 = time.time()
            print("All time:", time3 - time0)
            continue
        output_results = []
        for per_data in data[0]:
            score = per_data['score']
            if score < 0.5:
                continue
            bbox = per_data['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            if width > height:
                x1 = max(0, int(x1 / 640 * width))
                y1 = max(0, int(y1 / (640 - (diff / width * 640)) * height))
                x2 = min(width, int(x2 / 640 * width))
                y2 = min(height, int(y2 / (640 - (diff / width * 640)) * height))
            else:
                x1 = max(0, int(x1 / (640 - (diff / height * 640)) * width))
                y1 = max(0, int(y1 / 640 * height))
                x2 = min(width, int(x2 / (640 - (diff / height * 640)) * width))
                y2 = min(height, int(y2 / 640 * height))
            output_results.append([x1, y1, x2, y2, score])
        if not output_results:
            pipe.stdin.write(frame.tobytes())
            time3 = time.time()
            print("All time:", time3 - time0)
            continue
        output_results = np.array(output_results)
        online_targets = tracker.update(output_results)

        for t in online_targets:
            x, y, w, h = t.tlwh
            tid = t.track_id
            current_id.append(tid)
            label = "person" + " {}".format(tid)
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 6)
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            label_x = x
            label_y = y - 20 if y - 20 > label_height else y + 20
            cv2.rectangle(
                frame, (int(label_x), int(label_y - label_height)),
                (int(label_x + label_width), int(label_y + label_height)),
                (0, 0, 255), cv2.FILLED
            )
            cv2.putText(frame, label, (int(label_x), int(label_y + 5)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)
        pipe.stdin.write(frame.tobytes())
        time3 = time.time()
        print("All time:", time3 - time0)
        diff = set(current_id) - set(previous_id)
        if diff and set(current_id) != set(previous_id):
            previous_id += current_id
            current_id = []
            _, buffer = cv2.imencode('.jpg', origin_frame)
            origin_img_base64 = base64.b64encode(buffer).decode('utf-8')
            _, buffer = cv2.imencode('.jpg', frame)
            result_img_base64 = base64.b64encode(buffer).decode('utf-8')
            sio.emit("deploy",
                     json.dumps({"origin_img_base64": origin_img_base64, "result_img_base64": result_img_base64,
                                 "uuid": uuid}))

        previous_id += current_id
        current_id = []
    cap.release()
    pipe.stdin.close()
    pipe.wait()
    sio.emit("deploy", json.dumps({"message": "success"}))
    sio.disconnect()


def work(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    websocket_url = config['websocket_url']
    cmake_path = config['cmake_path']
    os.environ["PATH"] += os.pathsep + os.path.abspath(cmake_path)

    print("Init model...")
    person_deploy_handle = DeployHandle(
        lib_path=os.path.join(cmake_path, "AisDeployC.dll"),
        model_path="./models/person.aism",
        gpu_id=0,
        language="Chinese"
    )
    person_deploy_handle.model_init()
    print("Init success")

    sio.connect(websocket_url)
    print("WebSocket connection established.")
    src_url = config["src_url"]
    dst_url = config["dst_url"]
    uuid = config["uuid"]
    deploy(src_url, dst_url, uuid, person_deploy_handle)


if __name__ == "__main__":
    # config_path = sys.argv[1]
    config_path = "videoconfig.json"
    work(config_path)
