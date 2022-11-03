# hand-gesture-to-gamepad
This program uses a set of hand gestures taken by a webcam to act as an Xbox 360 controller.
The hand landmark recognition comes from [MediaPipe](https://google.github.io/mediapipe/solutions/hands.html) using code from [a translation](https://github.com/kinivi/hand-gesture-recognition-mediapipe) of [hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe). It incorporates the use of the [Virtual Gamepad](https://pypi.org/project/vgamepad/) library to emulate the controller.

To use, simply install the requirements, run the python file, calibrate, then use as you would an Xbox 360 controller on your computer.
This is a very early version I hope to improve in the future.

# Controls
- The standard hand placement has palms facing the camera.
- The pointer fingers on the left and right hand are used to control the left and right **joystick** respectively.
- Pressing Q while a hand is in frame will assign the neutral joystick position to the current pointer finger position, as shown by the blue circles on screen.
- Flipping the hand (ie turning your palm towards you) will cause the respective joystick button to be pushed.
- Pressing each finger to the tip of the thumb activates buttons as follows, from left to right
  - Left Hand
    - **Left bumper**
    - **Right bumper**
    - **Left trigger**
    - **Right trigger**
  - Right Hand
    - **A**
    - **B**
    - **X**
    - **Y**
- On the left hand, pressing the pointer finger to the base of the thumb activates the **back** button.
- On the right hand, pressing the pointer finger to the base of the thumb activates the **start** button.

# Settings
- Pressing W while the program is running will activate a dialouge to change the joystick deadzone, sensitivity, and radius on screen.

# Requirements
- MediaPipe 0.8.1
- OpenCV 3.4.2 or later
- Virtual GamePad 0.0.8 or later
