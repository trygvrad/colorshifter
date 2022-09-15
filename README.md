# Colorshifter

Colorshifter is an image manipulation app for transforming the colorspace of an image in L\*a\*b* color coordinates.

Open the app using loader.py

The left/right image is the original/transformed image

The histogram allows the user to edit the lightness (L\*)

The colormap on the right shows the a\*b\* plane, with a 2d histogram of the original image as white contour lines, and ditto for the transformed image as black. The shape/position/rotation of the rectangle defines how the original colorspace transforms onto the new colorspace.

Because not all colors in the transformed colorspace produce valid colors in RGB space, one may choose to limit the lighness or saturation for colors outside the RGB space.
To achieve specific transformations the a\*b\* plane may be rotated before the transformation occurs (rotating the primary axis of the rectangle)

The program used to edit a real image:
![](example_0.png?raw=true)

Used to recolor for prototyping:
![](example_2.png?raw=true)

Or used to recolor a texture (here also removing blue hues):
![](example_1.png?raw=true)

This is a hobby project that was made in a few days. Contributions are welcome.


The exe is compiled in python3.10 using pyinstaller 5.4.1 with the following packages/versions:

PySide2                   5.15.2.1, 
pyqtgraph                 0.12.4, 
numpy                     1.23.3,
contourpy                 1.0.5,
Pillow                    9.2.0,
scipy                     1.9.1,
colorspacious             1.1.2
