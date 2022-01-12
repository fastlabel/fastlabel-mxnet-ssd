# fastlabel-mxnet-ssd

## Setup

### Install libraries

```
$ pip install -r requirements.txt
```

### Clone original Face Recognition repository

```
$ git submodule init
$ git submodule update
```

## How to execute

### Prepare inputs data

Add downloaded model file(name must be `model.tar.gz`) to `./data/inputs/model/`.
Add images to `./data/inputs/images/`.

### Prediction

```
$ python main.py
```

### Tips

- `COLOR_PALETTE` is defined in `color_palette.py`. You can change it if you want.
- `CONFIDENCE_THRESHOLD` is defined in `main.py`. You can change it if you want.

## Output

### JSON

In `./data/outputs/`.

```json:predicts.json
[
  {
    "image": "data/inputs/images/pedestrian.png",
    "prediction": [ // [annotation index, confidence score, top-left x, top-left y, bottom-right x, bottom-right y point]
      [
        0.0, 0.6833887696266174, 237.33348083496094, 53.119441986083984,
        349.9132080078125, 350.6388244628906
      ],
      [
        0.0, 0.665876030921936, 18.011009216308594, 81.62196350097656,
        133.0487518310547, 311.8656005859375
      ]
    ]
  },
  ...
]
```

### Images

In `./data/outputs/images`.
The annotated images will be output.

### Model

In `./data/outputs/model`.
The unzipped files and the deployable model files will be output.
