# Sample Code 

- test.py

## Requirement
Please check packages in below are installed

- opencv3.x
- docopt

## usage

```
python test.py -f <video_file> -c <camera_pose_file>
```

For further information, try
```
python test.py --help
``` 

# Demo videos

Execution example

```
python test.py  -f data/samplevideo.avi -c data/camerapos.yml -d
```


## tracking result
![](https://raw.github.com/wiki/YoshiRi/IVPS/images/tracking.gif)

## xy position estimation
![](https://raw.github.com/wiki/YoshiRi/IVPS/images/plottraj.gif)