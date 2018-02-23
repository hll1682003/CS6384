# CS6384
Projects for Graduate Course CS6384 Computer Vision

**Synopsis**

The __project1__ contains __two__ programs, the first one do luminence *linear stretching*, and the second one do *histogram equalization*, both in *Luv* color space. These involve transform between *RGB* and *Luv* color space without invoking corresponding funcions in OpenCV library. They both take 6 additional arguments, the first four specify a rectangle window within the original image, the fifth one is the name of the input image, the sixth one is the name of the output image

For project1, the function __limit(x)__ is the last piece of the puzzle. Without it the output image will be flooded with annoying noise. 

**Prerequisites**

Title | Description
------------|-------------
Operating System | Windows10
Programming Language | Python 3.x
Library used | openCV, Numpy

**Installing**

After installing python 3 into OS, Run from the command line:
pip3 install opencv-python

**Explanation of decisions**
- Compared with the given program from the class, a red rectangle is added to outline the location of the window.
- In function __ ** XYZToLuv(x) ** __ , when X+Y+Z is zero, the function directly return a tuple [0,0,0] instead of calculating anything. Because when Y is zero, the pixel must be black which equals to a triple zero tuple. That's the way division by zero is handled.
- For the same reason, in function __ ** LuvToXYZ(x) ** __ : when the input argument x[0], which stands for L in Luv, equals to 0, then a tuple [0,0,0] can be directly obtained
- Another location that needs guard from division by zero is on line 167. When a rare situation which is the whole image has only one Luminance value, just don't do anything instead of trying to divide by the difference of the max and min Luminance.
- If chosen properly, both programs can improve image quality by adjusting the luminance, showing more details in dark and bright areas. However, when the Luminance within the window has a relatively small range compared with the whole image, the output image will look bad in the sense of having too much brightness or darkness.