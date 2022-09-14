# Colorshifter

Colorshifter is an image manipulation app for transforming the colorspace of an image in L*a*b* color coordinates.

The left/right image is the original/transformed image

The histogram allows the user to edit the lightness (L*)

The colormap on the right shows the a*b* plane, with a 2d histogram of the original image as white contour lines, and ditto for the transformed image as black. The shape/position/rotation of the rectangle defines how the original colorspace transforms onto the new colorspace.

Because not all colors in the transformed colorspace produce valid colors in RGB space, one may choose to limit the lighness or saturation for colors outside the RGB space.
To achieve specific transformations the a*b* plane may be rotated before the transformation occurs (rotating the primary axis of the rectangle)

The program used to edit a real image:
