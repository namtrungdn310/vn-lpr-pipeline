"""Fine-tune đúng checkpoint CCT-S-v1 bằng Keras/Torch của FastPlateOCR."""

import os
import sys
import types

import numpy as np

from ocr.config import save_json
from ocr.metrics import text_scores
from .common import settings, directory, read_json, freeze
from .sources import checkout, asset


def export_patch_extractor(model):
    """extract_patches → unfold tương đương để TorchScript giữ được shape kernel."""
    import torch

    def patches(layer, images):
        batch, height, width, channels = map(int, images.shape)
        size = int(layer.patch_size)
        values = torch.nn.functional.unfold(
            images.permute(0, 3, 1, 2), kernel_size=size, stride=size
        )
        return (
            values.reshape(batch, channels, size, size, -1)
            .permute(0, 4, 2, 3, 1)
            .reshape(batch, (height // size) * (width // size), size * size * channels)
        )

    for layer in model._flatten_layers():
        if type(layer).__name__ == "PatchExtractor":
            layer.call = types.MethodType(patches, layer)


def imports():
    os.environ["KERAS_BACKEND"] = "torch"
    source = checkout("cct")
    sys.path.insert(0, str(source))
    import keras
    from fast_plate_ocr.train.model import layers  # Đăng ký lớp Keras nguồn.
    from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
    from fast_plate_ocr.train.data.dataset import PlateRecognitionPyDataset

    return keras, load_plate_config_from_yaml, PlateRecognitionPyDataset


def train():
    import torch
    import albumentations as A

    keras, load_plate_config, Dataset = imports()
    if not torch.cuda.is_available():
        raise RuntimeError("CCT cần CUDA; không train CPU trên laptop")
    cfg = settings()
    keras.utils.set_random_seed(cfg["seed"])
    keras.mixed_precision.set_global_policy("float32")
    base = directory("CCT_S_VN")
    base.mkdir(parents=True, exist_ok=True)
    progress = base / "progress.json"
    state = (
        read_json(progress)
        if progress.exists()
        else {"epoch": 0, "best": [-1, -1], "wait": 0, "config": cfg}
    )
    if state["config"] != cfg:
        raise ValueError("Cấu hình khác lượt đang resume")
    if state.get("complete"):
        return
    plate = load_plate_config(asset("cct_s_v1_global_plate_config.yaml"))
    data = directory() / "data"
    augmentation = A.Compose(
        [A.RandomBrightnessContrast(p=0.5), A.Affine(scale=(0.9, 1.1), rotate=(-8, 8), p=0.5)]
    )
    train_data = Dataset(
        data / "train_cct.csv",
        plate,
        cfg["training"]["rec_batch"],
        transform=augmentation,
        shuffle=True,
        workers=0,
        use_multiprocessing=False,
    )
    val_data = Dataset(
        data / "val_cct.csv", plate, cfg["training"]["rec_batch"], shuffle=False, workers=0
    )
    if state["epoch"]:
        model = keras.models.load_model(base / "last.keras")
    else:
        native = keras.models.load_model(asset("cct_s_v1_global.keras"), compile=False)
        output = native.output["plate"] if isinstance(native.output, dict) else native.output
        model = keras.Model(native.inputs, {"plate": output})
        schedule = keras.optimizers.schedules.CosineDecay(
            cfg["training"]["lr"], cfg["training"]["epochs"] * len(train_data)
        )
        model.compile(
            optimizer=keras.optimizers.AdamW(
                schedule, weight_decay=cfg["training"]["weight_decay"], clipnorm=5
            ),
            loss={"plate": keras.losses.CategoricalCrossentropy(label_smoothing=0.01)},
        )

    class Checkpoint(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            predicted, references = [], []
            for i in range(len(val_data)):
                x, y = val_data[i]
                output = self.model.predict_on_batch(x)
                output = output["plate"] if isinstance(output, dict) else output
                for indices in np.asarray(output).reshape(-1, 9, len(plate.alphabet)).argmax(-1):
                    predicted.append(
                        "".join(plate.alphabet[int(j)] for j in indices).rstrip(plate.pad_char)
                    )
                for indices in y["plate"].argmax(-1):
                    references.append(
                        "".join(plate.alphabet[int(j)] for j in indices).rstrip(plate.pad_char)
                    )
            scores = text_scores(predicted, references)
            criterion = [scores["ACC"], -scores["CER"]]
            state["wait"] += 1
            if criterion > state["best"]:
                state.update(best=criterion, wait=0, best_val=scores, best_epoch=epoch + 1)
                self.model.save(base / "best.keras")
            self.model.save(base / "last.keras")
            state["epoch"] = epoch + 1
            save_json(progress, state)
            print(
                f"CCT epoch {epoch+1}: val ACC={scores['ACC']:.2f} CER={scores['CER']:.2f}",
                flush=True,
            )
            if state["wait"] >= cfg["training"]["patience"]:
                self.model.stop_training = True

    model.fit(
        train_data,
        initial_epoch=state["epoch"],
        epochs=cfg["training"]["epochs"],
        validation_data=val_data,
        callbacks=[keras.callbacks.TerminateOnNaN(), Checkpoint()],
        verbose=2,
    )
    if not (base / "best.keras").is_file():
        raise ValueError("CCT chưa tạo best trên val")
    state["complete"] = True
    save_json(progress, state)


def export():
    import onnxruntime as ort
    from ocr.data import image
    from .preprocessing import cct_input
    from .data import rows

    keras, _, _ = imports()
    base = directory("CCT_S_VN")
    state = read_json(base / "progress.json")
    if not state.get("complete"):
        raise ValueError("Chưa train xong CCT")
    import torch

    (base / "onnx").mkdir(exist_ok=True)
    target = base / "onnx/model.onnx"
    checks = []
    with keras.device("cpu"):
        model = keras.models.load_model(base / "best.keras", compile=False).cpu()
        samples = [(row, cct_input(image(row["path"]))) for row in rows("val")[:3]]
        with torch.inference_mode():
            reference_native = [
                np.asarray(model.predict_on_batch(data)["plate"]) for row, data in samples
            ]
        export_patch_extractor(model)

        class ExportModel(torch.nn.Module):
            def __init__(self, native):
                super().__init__()
                self.native = native

            def forward(self, x):
                output = self.native(x, training=False)
                return output["plate"] if isinstance(output, dict) else output

        exported = ExportModel(model).eval()
        sample = torch.zeros((1, 64, 128, 3), dtype=torch.float32)
        torch.onnx.export(
            exported,
            sample,
            str(target),
            input_names=["input"],
            output_names=["plate"],
            opset_version=17,
            dynamo=False,
        )
        engine = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
        for (row, data), native in zip(samples, reference_native):
            with torch.inference_mode():
                reference = exported(torch.from_numpy(data)).numpy()
            np.testing.assert_allclose(reference, native, rtol=1e-5, atol=1e-6)
            actual = engine.run(None, {engine.get_inputs()[0].name: data})[0]
            np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=1e-4)
            checks.append(
                {"image": row["relative"], "max_abs": float(np.max(np.abs(actual - reference)))}
            )
    save_json(
        base / "onnx/runtime.json",
        {
            "family": "cct",
            "chars": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_",
            "input_dtype": "float32",
        },
    )
    return freeze(
        "CCT_S_VN",
        {
            "initialization": settings()["cct_initialization"],
            "best_val": state["best_val"],
            "source_commit": settings()["sources"]["cct"]["revision"],
            "conversion_checks": checks,
        },
    )
