import os
import subprocess
import glob
import tarfile
import json
import time
from collections import namedtuple

import cv2
import numpy as np
import mxnet as mx

from color_palette import COLOR_PALETTE


CONFIDENCE_THRESHOLD = 0.4
OUTPUT_ANNOTATED = True  # Output annotated images if True


INPUT_FOLDER_PATH = os.path.join("data", "inputs")
INPUT_IMAGE_FOLDER_PATH = os.path.join(
    INPUT_FOLDER_PATH, "images")
INPUT_MODEL_FILE_PATH = os.path.join(
    INPUT_FOLDER_PATH, "model", "model.tar.gz")
OUTPUT_FOLDER_PATH = os.path.join("data", "outputs")
OUTPUT_MODEL_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, "model")
OUTPUT_IMAGES_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, "images")
OUTPUT_PREDICTS_FILE_PATH = os.path.join(OUTPUT_FOLDER_PATH, "predicts.json")
os.makedirs(OUTPUT_MODEL_FOLDER_PATH, exist_ok=True)
os.makedirs(OUTPUT_IMAGES_FOLDER_PATH, exist_ok=True)


def __update_mxnet_submodule():
    # To prevent an error below.
    # File "/xxx/fastlabel-python-sdk/fastlabel/mxnet-ssd/symbol/symbol_builder.py", line 2, in <module>
    #     from common import multi_layer_feature, multibox_layer
    # ModuleNotFoundError: No module named "common"
    os.environ["PYTHONPATH"] = os.path.dirname(
        os.path.abspath(__file__)) + "/mxnet-ssd/symbol/"
    print("export PYTHONPATH={}".format(os.environ.get("PYTHONPATH")))

    # Modify "xrange" to "range" in mxnet-ssd/symbol/symbol_factory.py because xrange is abolished in Python3
    file_name = os.path.dirname(os.path.abspath(
        __file__)) + "/mxnet-ssd/symbol/symbol_factory.py"
    with open(file_name) as f:
        data_lines = f.read()
    data_lines = data_lines.replace("xrange", "range")
    with open(file_name, mode="w") as f:
        f.write(data_lines)


def __get_ctx():
    try:
        gpus = mx.test_utils.list_gpus()
        if len(gpus) > 0:
            ctx = []
            for gpu in gpus:
                ctx.append(mx.gpu(gpu))
        else:
            ctx = [mx.cpu()]
    except:
        ctx = [mx.cpu()]
    return ctx


def __get_deployable_model():
    hyperparams_json_path = os.path.join(
        OUTPUT_MODEL_FOLDER_PATH, "hyperparams.json")
    with open(hyperparams_json_path, "r") as f:
        hyperparams = json.load(f)

    network = "resnet50" if hyperparams["base_network"] == "resnet-50" else hyperparams["base_network"]
    subprocess.run(["python3", os.path.dirname(os.path.abspath(__file__)) + "/mxnet-ssd/deploy.py", "--network", network,
                    "--num-class", hyperparams["num_classes"], "--nms", hyperparams["nms_threshold"], "--data-shape", hyperparams["image_shape"], "--prefix", os.path.join(OUTPUT_MODEL_FOLDER_PATH, "model_algo_1")])

    shape = int(hyperparams["image_shape"])

    ctx = __get_ctx()[0]
    print(f"ctx: {ctx}")

    input_shapes = [("data", (1, 3, shape, shape))]

    param_path = os.path.join(OUTPUT_MODEL_FOLDER_PATH, "deploy_model_algo_1")
    print(f"param_path: {param_path}")

    sym, arg_params, aux_params = mx.model.load_checkpoint(param_path, 0)
    mod = mx.mod.Module(symbol=sym, label_names=[], context=ctx)
    mod.bind(for_training=False, data_shapes=input_shapes)
    mod.set_params(arg_params, aux_params)

    return mod, shape


def __predict(mod, reshape: tuple[int, int], image_file_path: str):
    # Switch RGB to BGR format (which ImageNet networks take)
    img = cv2.imread(image_file_path)
    org_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        return None

    # Resize image to fit network input
    img = cv2.resize(img, reshape)

    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]

    Batch = namedtuple("Batch", ["data"])
    mod.forward(Batch([mx.nd.array(img)]))
    predicts = mod.get_outputs()[0].asnumpy()
    predicts = np.squeeze(predicts)

    predicts = predicts[predicts[:, 0] != -1]
    predicts = predicts[predicts[:, 1]
                        >= CONFIDENCE_THRESHOLD]

    h, w = org_img.shape[:2]
    predicts[:, (2, 4)] *= w
    predicts[:, (3, 5)] *= h

    return predicts, org_img


def main():

    with tarfile.open(INPUT_MODEL_FILE_PATH) as tar:
        tar.extractall(OUTPUT_MODEL_FOLDER_PATH)

    __update_mxnet_submodule()
    mod, shape = __get_deployable_model()

    image_file_paths = glob.glob(os.path.join(INPUT_IMAGE_FOLDER_PATH, "*"))
    if not image_file_paths:
        raise Exception(
            "Folder does not have any file.", 422)

    to_writes = []
    with open(OUTPUT_PREDICTS_FILE_PATH, "w") as outfile:
        for image_file_path in image_file_paths:
            if not image_file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            predicts, org_img = __predict(mod, (shape, shape), image_file_path)
            to_write = {"image": image_file_path,
                        "prediction": predicts.tolist()}
            to_writes.append(to_write)

            if OUTPUT_ANNOTATED:
                for predict in predicts:
                    (anno_idx, confidence_score, x1, y1, x2, y2) = predict
                    color = COLOR_PALETTE[int(
                        anno_idx+1)*3:int(anno_idx+1)*3 + 3]
                    if not color:
                        color = (0, 0, 0)
                        print(
                            "Due to lack of color, drawing uses black. Please add color in COLOR_PALETTE.")
                    cv2.rectangle(org_img, [int(x1), int(y1)], [int(x2), int(y2)],
                                  color=color, thickness=2)
                    cv2.imwrite(os.path.join(OUTPUT_IMAGES_FOLDER_PATH,
                                             os.path.basename(image_file_path)), org_img)
        outfile.write(json.dumps(to_writes))


if __name__ == "__main__":
    start = time.time()
    main()
    t = time.time() - start
    print(f"elapsed time: {t} (s)")
