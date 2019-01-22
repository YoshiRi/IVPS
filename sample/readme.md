# Sample Code 

- test.py : tracking and show figure before quitting
- realtime_output.py : tracking and print the marker position in realtime

## Requirement
Please check packages in below are installed

- opencv2.7
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

- Execution example (with video file)

```
python test.py  -f data/samplevideo.avi -c data/camerapos.yml -d
```

- Execution example (with usb camera input)

If you want to get realtime position output, try following command.

```
python realtime_output.py -d
```

You can update the print function in l.103 with your own output function.


## tracking result
![](https://raw.github.com/wiki/YoshiRi/IVPS/images/tracking.gif)

## xy position estimation
![](https://raw.github.com/wiki/YoshiRi/IVPS/images/plottraj.gif)