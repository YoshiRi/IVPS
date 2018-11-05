# IVPS
Indoor Vision based Positioning System (Under Construction)

# Setting

## Requirement

- Assuming ROS Kinetic Environment on Ubuntu
- Python2.7 with OpenCV3.X with Opencv-contrib package

## Usage

```python
import cv2
import numpy as np
import sys
import math

sys.path.append("../")

from vmarker import *
```


# Real Setting

- Put marker on the ground 
- Set a ground coordinate and measure marker position

Remember opencv using left-handed coordinates!