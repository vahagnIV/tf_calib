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
Our goal is to find the camera matrix K together with distortion coefficients, the so-called intrinsic parameters of the camera.

The camera matrix reads:
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?K%3D%5Cleft%28%5Cbegin%7Barray%7D%7Bccc%7D%0A%20%20f_x%20%26%200%20%26%20c_x%20%5C%5C%0A%20%200%20%26%20f_y%20%26%20c_y%20%5C%5C%0A%20%200%20%26%200%20%26%201%0A%20%5Cend%7Barray%7D%5Cright%29"/>, 
 </p>
 
where f<sub>x</sub>, f<sub>y</sub> are focal lengths of the camera and c<sub>x</sub>, c<sub>y</sub> are the pixel coordinates of the principal point.

The distortion coefficients depend on the model. Currently we have implemented only simple radial distortion coefficients k<sub>1</sub>, k<sub>2</sub>, k<sub>3</sub> from <a href="https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html">OpenCV docs</a>. 

The world coordinates (x<sub>1</sub>,x<sub>2</sub>,x<sub>3</sub>) in the camera coordinate system are related to the pixel coordinates  (&xi;<sub>1</sub>,&xi;<sub>2</sub>,1) via the camera matrix as follows:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?s%5Cxi_%5Calpha%20%3D%5Csum%5Climits_%7B%5Cbeta%20%3D%201%7D%5E3%20K_%7B%5Calpha%5Cbeta%7Dx_%5Cbeta%2C%5Cquad%20%5Calpha%3D1%2C2%2C3" /> 
</p>

with a scale parameter s, which corresponds to the arbitrary scale due to the fact that the camera projects an entire ray to a single point.

The inverse of this relation looks as follows:
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?x_%5Calpha%20%3D%20%5Csum%5Climits_%7B%5Cbeta%3D1%7D%5E3K%5E%7B-1%7D_%7B%5Calpha%5Cbeta%7D%5Cxi_%5Cbeta%20s%20%5Cequiv%20%20su_%5Calpha%2C%5Cquad%20%5Calpha%3D1%2C2%2C3"/>       (1)
</p>

The vectors u<sup>i</sup> define the ray corresponding to the i-th point.

### Minimization

Let c<sup>i</sup><sub>&alpha;</sub> be the coordinates of a calibration pattern in its own coordinate system. Here and further the latin index i runs from 1 to N, where N is the number of points in the calibration pattern. Greek indices &alpha;,&beta;... run from 1 to 3. Note that we do not assume anything about the structure of the calibration pattern.

Now, let &xi;<sup>i</sup><sub>&alpha;</sub> be the pixel coordinates of the ith pattern point on the image.

Assuming the camera intrinsic parameters are known we can find a rotation matrix R and a translation vector T which, together with the scale parameters s<sup>i</sup>,  minimize the following error:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?E%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5EN%5Csum_%7B%5Calpha%3D1%7D%5E3%5Cleft%28x%5Ei%20-%20%5Chat%7BR%7Dc%5Ei%20%2B%20T%20%5Cright%29%5E2%5Cequiv%20%5Csum%5Climits_%7Bi%3D1%7D%5EN%5Csum_%7B%5Calpha%3D1%7D%5E3%5Cleft%28u%5Ei_%5Calpha%20s%5Ei%20-%20%5Chat%7BR%7Dc%5Ei_%5Calpha%20%2B%20T_%5Calpha%20%5Cright%29%5E2"/>
</p>

The quantities *u* are defined in (1). We equipped the variables from the previous section with additional index i, which indicates the number of the point in the calibration pattern.

Indeed, sequientially taking the derrivatives of *E* by *s<sup>i</sup>*, *T*, and *R* and equaling them to zero we obtain the following expressions for (1):
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?E%3D%5Csum_%7Bi%2Cj%3D1%7D%5EN%20%5Csum_%7B%5Calpha%2C%5Cbeta%3D1%7D%5E3%20%20A%5Ei_%7B%5Calpha%7DU%5E%7Bij%7D_%7B%5Calpha%5Cbeta%7DA%5Ej_%5Cbeta"/>  (2)
</p>

Here we defined the following auxilliary quantities:

The projector alogn u<sup>i</sup>
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?%5COmega%5Ei_%7B%5Calpha%5Cbeta%7D%3D%5Cdelta_%7B%5Calpha%5Cbeta%7D%20-%20%5Cfrac%7Bu%5Ei_%5Calpha%20u%5Ei_%5Cbeta%7D%7B%28u%5Eiu%5Ei%29%7D"/>,  
</p>

The projection of the rotated vector Rc<sup>i</sup> on the line u<sup>i</sup>

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?A%5Ei%3D%5COmega%5Ei%20R%20c%5Ei"/>, 
</p>

The sum of projectors and its inverse:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?%5COmega%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5EN%20%5COmega%5Ei%2C%5Cquad%20W%3D%5COmega%5E%7B-1%7D"/>, 
</p>

With this notations the matrix *U* looks:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?U%3D-W%5Cotimes%20%5Cmathbb%7BJ%7D%5EN%20+%20%5Cmathbb%7BI%7D_3%5Cotimes%5Cmathbb%7BI%7D%5EN"/>,       (3)
</p>

where *J* <sup>N</sup> is NxN all-one matrix matrix and *I* 's are identity matrices of the corresponding dimensions.

The definition (3) in the operator notations reads:


<p align="center">
<img src="https://latex.codecogs.com/gif.latex?U%5E%7Bij%7D_%7B%5Calpha%5Cbeta%7D%3D-W_%7B%5Calpha%5Cbeta%7D%20%2B%20%5Cdelta%5E%7Bij%7D%5Cdelta_%7B%5Calpha%5Cbeta%7D"/>  
</p>


The rotation matrix is determined from the following equation:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?R%5ETM%3DM%5ETR"/>  (4)
</p>

where the matrix M reads:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?M_%7B%5Calpha%5Cbeta%7D%3D%5Csum_%7Bi%2Cj%3D1%7D%5EN%20%5Csum_%7B%5Cgamma%2C%5Cdelta%3D1%7D%5E3%20A%5Ei_%5Cdelta%20U%5E%7Bij%7D_%7B%5Cdelta%5Cgamma%7D%5COmega%5Ej_%7B%5Cgamma%5Cbeta%7Dc%5Ej_%5Calpha"/>  
</p>

To summarize, the algorithm for determining the parameters *K* is the following:

1. Initialize K
2. For each image 
   1. Solve the equation (4) subject to R<sup>T</sup>R = I
   2. Apply gradients descent to parameters *K* with loss function (2)


