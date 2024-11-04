from track.byte_tracker import BYTETracker
import cv2
from modules.deploy_handle import DeployHandle
import time
import numpy as np
import os
os.environ["PATH"] += os.pathsep + os.path.abspath("./cmake-build-release")


class Args:
    def __init__(self):
        self.mot20 = False
        self.track_thresh = 0.5
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.frame_rate = 30


args = Args()
tracker = BYTETracker(args)
deploy_handle = DeployHandle(
    lib_path="./cmake-build-release/AisDeployC.dll",
    model_path="./models/" + "vehicle.aism",
    gpu_id=1,
    language="Chinese"
)
deploy_handle.model_init()

cap = cv2.VideoCapture("car_school.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(fps, width, height)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

previous_id = []
current_id = []
count = 0
while cap.isOpened():
    time0 = time.time()
    ret, frame = cap.read()

    if not ret:
        break
    output_results = []
    result = deploy_handle.deploy(frame)
    data = result['data']
    for per_data in data[0]:
        score = per_data['score']
        if score < 0.5:
            continue
        bbox = per_data['bbox']
        label = per_data['category']

        # 将单个物体的数据添加到output_results中
        output_results.append([bbox[0], bbox[1], bbox[2], bbox[3], score])
    output_results = np.array(output_results)

    online_targets = tracker.update(output_results)
    for t in online_targets:
        x, y, w, h = t.tlwh
        tid = t.track_id
        current_id.append(tid)
        label = "car" + " {}".format(tid)
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
    out.write(frame)
    diff = set(current_id) - set(previous_id)
    if diff and set(current_id) != set(previous_id):
        count +=1
        print(count)
        cv2.imwrite(f'{count}.jpg', frame)
    previous_id += current_id
    current_id = []
cap.release()
