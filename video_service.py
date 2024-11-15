import asyncio
import websockets
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
import ffmpeg


class Args:
    def __init__(self):
        self.mot20 = False
        self.track_thresh = 0.5
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.frame_rate = 30


args = Args()
frame_queue = asyncio.Queue(maxsize=100000)
result_queue = asyncio.Queue(maxsize=100000)
batch_size = 1


async def send_ping(websocket):
    while True:
        await websocket.send(json.dumps({"type": "ping"}))
        await asyncio.sleep(2)  # 每2秒发送一次心跳


async def deploy(person_deploy_handle, height, width, uuid, websocket):
    previous_id = []
    current_id = []
    send_ping_task = asyncio.create_task(send_ping(websocket))
    tracker = BYTETracker(args)
    while True:
        input_json = {"data_list": []}
        batch_frames = []
        while len(batch_frames) < batch_size:
            if frame_queue.empty():
                await asyncio.sleep(0.01)  # 队列为空时，可以稍微延迟一会儿
                continue
            print("size of batch_frames is:", len(batch_frames))
            frame = await frame_queue.get()
            batch_frames.append(frame)
            input_json = {"data_list": []}
            for deploy_img in batch_frames:
                diff = abs(height - width)
                if height > width:
                    pad_img = cv2.copyMakeBorder(deploy_img, 0, 0, 0, diff, cv2.BORDER_CONSTANT, 0)
                else:
                    pad_img = cv2.copyMakeBorder(deploy_img, 0, diff, 0, 0, cv2.BORDER_CONSTANT, 0)

                img_rs = cv2.resize(pad_img, (640, 640))
                encoded_image = cv2.imencode(".jpg", img_rs.copy())[1].tobytes()  # bytes类型
                qrcode = base64.b64encode(encoded_image).decode()
                file_json = {"type": "base64", "data": qrcode, "ch": 3}
                input_json["data_list"].append(file_json)
        time0 = time.time()

        start_deploy_time = time.time()
        result = person_deploy_handle.deployjson(input_json)
        end_deploy_time = time.time()
        print("deploy time:", end_deploy_time - start_deploy_time)
        print(result)
        batch_data = result['data']
        for i, data in enumerate(batch_data):
            if data is None:
                await result_queue.put(batch_frames[i])
                continue
            output_results = []
            image = batch_frames[i].copy()
            for per_data in data:
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
                await result_queue.put(batch_frames[i])
                continue
            time_ts = time.time()
            output_results = np.array(output_results)
            online_targets = tracker.update(output_results)
            time_te = time.time()
            print("track time:", time_te - time_ts)
            image = batch_frames[i].copy()
            time_ds = time.time()
            for t in online_targets:
                x, y, w, h = t.tlwh
                tid = t.track_id
                current_id.append(tid)
                label = "person" + " {}".format(tid)
                cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 6)
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                label_x = x
                label_y = y - 20 if y - 20 > label_height else y + 20
                cv2.rectangle(
                    image, (int(label_x), int(label_y - label_height)),
                    (int(label_x + label_width), int(label_y + label_height)),
                    (0, 0, 255), cv2.FILLED
                )
                cv2.putText(image, label, (int(label_x), int(label_y + 5)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)
            time_de = time.time()
            print("draw time:", time_de - time_ds)
            await result_queue.put(image)
            diff_id = set(current_id) - set(previous_id)
            time_bs = time.time()
            if diff_id and set(current_id) != set(previous_id):
                previous_id += current_id
                current_id = []
                _, buffer = cv2.imencode('.jpg', batch_frames[i])
                origin_img_base64 = base64.b64encode(buffer).decode('utf-8')
                _, buffer2 = cv2.imencode('.jpg', image)
                result_img_base64 = base64.b64encode(buffer2).decode('utf-8')
                await websocket.send(
                    json.dumps(
                        {"origin_img_base64": origin_img_base64,
                         "result_img_base64": result_img_base64,
                         "uuid": uuid}
                    )
                )
            time_be = time.time()
            print("send time:", time_be - time_bs)
            previous_id += current_id
            current_id = []
        time1 = time.time()
        print("All time:", time1 - time0)
        del batch_frames
        await asyncio.sleep(0)


async def pull_frames(src_url, process1, width, height):
    loop = asyncio.get_running_loop()
    while True:
        in_bytes = await loop.run_in_executor(None, process1.stdout.read, width * height * 3)
        if not in_bytes:
            print("bytes is None")
            process1.terminate()
            process1.wait()
            process1 = (
                ffmpeg
                .input(
                    src_url
                )
                .filter('fps', fps=12, round='up')
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .overwrite_output()
                .run_async(pipe_stdout=True)
            )
            await asyncio.sleep(0.1)
            continue
        print("pull success")
        in_frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
        in_frame = in_frame[:, :, [2, 1, 0]]
        await frame_queue.put(in_frame)
        print("put success, size of frame_queue is ", frame_queue.qsize())
        await asyncio.sleep(0)


async def push_frames(process2):
    while True:
        if result_queue.empty():
            await asyncio.sleep(0.1)
            continue

        out_frame = await result_queue.get()
        out_frame = out_frame[:, :, [2, 1, 0]]
        process2.stdin.write(
            out_frame
            .astype(np.uint8)
            .tobytes()
        )
        print("push success")
        await asyncio.sleep(0)


async def work(src_url, dst_url, uuid, cmake_path, person_deploy_handle, websocket):
    probe = ffmpeg.probe(src_url)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])

    avg_frame_rate = video_stream.get('avg_frame_rate', '0/1')

    if avg_frame_rate == '0/0':
        avg_frame_rate = '30/1'

    print(avg_frame_rate)
    frame_rate = int(eval(avg_frame_rate))
    print("[DEBUG]------------------- width, height, fps------------------:", width, height, frame_rate)
    process1 = (
        ffmpeg
        .input(
            src_url
        )
        .filter('fps', fps=12, round='up')
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .overwrite_output()
        .run_async(pipe_stdout=True)
    )

    process2 = (
        ffmpeg
        .input("-", format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), hwaccel='cuda')
        .filter('setpts', 'PTS*({}/12)'.format(frame_rate))
        .output(dst_url, vcodec='libx264', pix_fmt='yuv420p', format='flv', b='2M')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    await asyncio.gather(
        pull_frames(src_url, process1, width, height),
        deploy(person_deploy_handle, height, width, uuid, websocket),
        push_frames(process2)
    )


async def main(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    websocket_url = config['websocket_url']

    cmake_path = config['cmake_path']
    os.environ["PATH"] += os.pathsep + os.path.abspath(cmake_path)
    print("Init model...")
    person_deploy_handle = DeployHandle(
        lib_path=os.path.join(cmake_path,"AisDeployC.dll"),
        model_path="./models/person.aism",
        gpu_id=0,
        language="Chinese"
    )
    person_deploy_handle.model_init()
    print("Init success")

    async with websockets.connect(websocket_url) as websocket:
        print("WebSocket connection established.")
        src_url = config["src_url"]
        dst_url = config["dst_url"]
        uuid = config["uuid"]
        debug = config["debug"]
        if not debug:
            sys.stdout = open(os.devnull, 'w')
        await work(src_url, dst_url, uuid, cmake_path, person_deploy_handle, websocket)


if __name__ == "__main__":
    config_path = sys.argv[1]
    # config_path = "videoconfig.json"
    asyncio.run(main(config_path))
