import torch


def download_model(model_name, input_shape=(1, 3, 224, 224), pretrained=True):
    try:
        model = torch.hub.load("pytorch/vision:v0.10.0", model_name, pretrained=pretrained)
        model_path = f'onnx/{model_name}.onnx'

        torch.onnx.export(
            model,
            torch.randn(*input_shape),
            model_path,
            verbose=False,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}}
        )

        print(f"Torch model {model_name} successfully exported to {model_path}")
    except Exception as e:
        print(f"Ошибка при загрузке или экспорте модели {model_name}: {e}")


models = [
    ('alexnet', (1, 3, 224, 224)),
    ('vgg16', (1, 3, 224, 224)),
    ('vgg16_bn', (1, 3, 224, 224)),
    ('mobilenet_v2', (1, 3, 224, 224)),
    ('squeezenet1_1', (1, 3, 224, 224)),
    ('mnasnet0_5', (1, 3, 224, 224))
    ('resnet50', (1, 3, 224, 224))
]

if __name__ == "__main__":
    for model in models:
        print(f'downloading {model[0]}')
        download_model(model[0], model[1])
