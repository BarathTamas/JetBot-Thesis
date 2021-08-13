# JetBot Thesis

### Road following demo
Video of road following with stop/move classification from the POV of the JetBot. Video is at ~3x real speed. At the sensor stops (green rectangles) the robot performs measurements for 10 second without processing images, hence the short "cuts".
In every frame, the JetBot has to 1. estimate where to go based on the marking tape (prediction marked with a red circle), 2. decide whether a stop has been reached. The robot is using a multi-tasking CNN (with EfficientNet b0 as the encoder), jointly retrained for the regression and classification tasks.

https://user-images.githubusercontent.com/44137494/129424808-45d1cc8e-4523-45ef-bdb2-94560483d9dd.mp4
