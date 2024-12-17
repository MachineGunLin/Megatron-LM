import os
import json
from glob import glob
import shutil
from tqdm import tqdm

json_paths = glob("demo_data/*.jsonl")


for json_path in json_paths:
    lines_removed = []
    with open(json_path) as f:
        for line in tqdm(f.readlines()):

            data_dict = json.loads(line)
            input_msgs = data_dict["conversations"]

            for msg in input_msgs:
                content = msg['content']
                if isinstance(content, dict):
                    if "image" in content:
                        if isinstance(content["image"], str):
                            src_path = content["image"]
                            tgt_path = f"audio_image_data/{os.path.basename(src_path)}"
                            shutil.copy(src_path, tgt_path)
                            tgt_path = os.path.join("output", tgt_path)
                            content["image"] = tgt_path
                        if isinstance(content["image"], list):
                            paths = []
                            for src_path in content["image"]:
                                tgt_path = f"audio_image_data/{os.path.basename(src_path)}"
                                shutil.copy(src_path, tgt_path)
                                tgt_path = os.path.join("output", tgt_path)
                                paths.append(tgt_path)
                                content["image"] = paths

                    if "audio" in content and content["audio"] is not None:
                        src_path = content["audio"]
                        tgt_path = f"audio_image_data/{os.path.basename(src_path)}"
                        shutil.copy(src_path, tgt_path)
                        tgt_path = os.path.join("output", tgt_path)
                        content["audio"] = tgt_path
            lines_removed.append(json.dumps(data_dict))
    with open(json_path.replace("demo_data", "demo_data_fix"), "w") as f:
        for line in lines_removed:
            f.write(line + '\n')

