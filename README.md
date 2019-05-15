# Tensorflow calibrator
Tensorflow camera calibrator

## Prerequisites

```bash
sudo pip3 install numpy, scipy, tensorflow_gpu
```

The example data is collected using my Logitech C 310 webcam.

After 10 epochs we reach precision < 10<sup>-2</sup> size of the quare (~7cm)

## Theory

### Pinhole camera
Our goal is to find the camera matrix K together with distortion coefficient, the so-called intrinsic parameters of the camera.

The camera matrix reads:
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?K%3D%5Cleft%28%5Cbegin%7Barray%7D%7Bccc%7D%0A%20%20f_x%20%26%200%20%26%20c_x%20%5C%5C%0A%20%200%20%26%20f_y%20%26%20c_y%20%5C%5C%0A%20%200%20%26%200%20%26%201%0A%20%5Cend%7Barray%7D%5Cright%29"/>, 
 </p>
 
where f<sub>x</sub>, f<sub>y</sub> are focal lengths of the camera and c<sub>x</sub>, c<sub>y</sub> are the pixel coordinates of the optical center.

The distortion coefficients depend on the model. Currently we have implemented only simple distortion coefficients k<sub>1</sub>, k<sub>2</sub>, k<sub>3</sub> from <a href="https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html">OpenCV docs</a>. 

The world coordinates (x<sub>1</sub>,x<sub>2</sub>,x<sub>3</sub>) in the camera coordinate system are related to the pixel coordinates  (&xi;<sub>1</sub>,&xi;<sub>2</sub>,1) via the camera matrix as follows:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?s%5Cxi_%5Calpha%20%3D%5Csum%5Climits_%7B%5Cbeta%20%3D%201%7D%5E3%20K_%7B%5Calpha%5Cbeta%7Dx_%5Cbeta%2C%5Cquad%20%5Calpha%3D1%2C2%2C3" /> 
</p>

with a scale parameter s, which corresponds to the arbitrary scale due to the fact that the camera projects an entire ray to a single point.

The inverse of this relation looks as follows:
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?x_%5Calpha%20%3D%20%5Csum%5Climits_%7B%5Cbeta%3D1%7D%5E3K%5E%7B-1%7D_%7B%5Calpha%5Cbeta%7D%5Cxi_%5Cbeta%20s%20%5Cequiv%20%20su_%5Calpha%2C%5Cquad%20%5Calpha%3D1%2C2%2C3"/>       (1)
</p>

### Minimization

Let c<sup>i</sup><sub>&alpha;</sub> be the coordinates of a pattern in its own coordinate system. Here and further the latin index i runs from 1 to N, where N is the number of points in the calibration pattern. greek indices &alpha;&beta;... run from 1 to 3. 

Now, let &xi;<sup>i</sup><sub>&alpha;</sub> be the pixel coordinate of the corresponding pattern point on the image.

Assuming the camera intrinsic parameters are known we can find a rotation matrix R and a translation vector T which , together with the scale parameters s<sup>i</sup>,  minimize the following error:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?E%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5EN%5Csum_%7B%5Calpha%3D1%7D%5E3%5Cleft%28x%5Ei%20-%20%5Chat%7BR%7Dc%5Ei%20%2B%20T%20%5Cright%29%5E2%5Cequiv%20%5Csum%5Climits_%7Bi%3D1%7D%5EN%5Csum_%7B%5Calpha%3D1%7D%5E3%5Cleft%28u%5Ei_%5Calpha%20s%5Ei%20-%20%5Chat%7BR%7Dc%5Ei_%5Calpha%20%2B%20T_%5Calpha%20%5Cright%29%5E2"/>
</p>

The quantities *u* are defined in (1). We equipped the variables from the previous section with additional index i, which enumerates the points on the calibration pattern.

Indeed, sequientially taking the derrivatives of *E* by *s<sup>i</sup>*, *T*, and *R* and equaling them to zero we obtain the following expressions for (1):
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?E%3D%5Csum_%7Bi%2Cj%3D1%7D%5EN%20%5Csum_%7B%5Calpha%2C%5Cbeta%3D1%7D%5E3%20%20A%5Ei_%7B%5Calpha%7DU%5E%7Bij%7D_%7B%5Calpha%5Cbeta%7DA%5Ej_%5Cbeta"/>
</p>

Here we defined the following auxilliary quantities:
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?%5COmega%5Ei_%7B%5Calpha%5Cbeta%7D%3D%5Cdelta_%7B%5Calpha%5Cbeta%7D%20-%20%5Cfrac%7Bu%5Ei_%5Calpha%20u%5Ei_%5Cbeta%7D%7B%28u%5Eiu%5Ei%29%7D"/>,  <img src="https://latex.codecogs.com/gif.latex?A%5Ei%3D%5COmega%5Ei%20R%20c%5Ei"/>, 
</p>

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?E%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5EN%5Csum_%7B%5Calpha%3D1%7D%5E3%5Cleft%28x%5Ei%20-%20%5Chat%7BR%7Dc%5Ei%20%2B%20T%20%5Cright%29%5E2%5Cequiv%20%5Csum%5Climits_%7Bi%3D1%7D%5EN%5Csum_%7B%5Calpha%3D1%7D%5E3%5Cleft%28u%5Ei_%5Calpha%20s%5Ei%20-%20%5Chat%7BR%7Dc%5Ei_%5Calpha%20%2B%20T_%5Calpha%20%5Cright%29%5E2"/>
</p>



