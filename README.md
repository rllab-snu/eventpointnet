# EventPointNet: Supervised keypoint detector with neuromorphic camera data
This repository provides python implementation of supervised keypoint detector with neuromorphic camera data, known as EventPointNet. 

## How to run
Argument for the method is as follows

| Argument | Abbreviation | Description | Option |
|---|:---:|:---:|:---:|
|`--width`|`-W`| Resize width (Optional) |default = `None`|
|`--height`|`-H`| Resize height (Optional) |default = `None`|

<br>

With the arguments above, you can run the code by executing `run.py`.

```bash
python run.py --width <WIDTH> --height <HEIGHT> 
```

## Dependencies

```
python 3.8.5
pytorch 1.7.1
opencv 4.5.2
scikit-image 0.18.1
numpy 1.19.2
```

Aside from above dependencies, docker container for this repository is provided [here](https://hub.docker.com/r/rllabsnu/eventpointnet)
