# AI
A repo containing mainly some my implementations of Deep Learning models.


## Usage:
All of them are successfully tested to run in Anaconda Spyder on both Ubuntu 18 and Windows 10 with the following packages:

* Tensorflow (gpu)          (for backend. It has cudNN dependency in my case, so maybe cudNN with some bindings will be needed, too.)

* Keras                     (for simplification of Tensorflow. Less code does essentially the same 
                             + Keras automatically runs on the GPU version of Tensorflow, if that one is installed.
                             Another plus is that data augmentation is executed in a parallelised manner.) 

* Numpy                     (for speed-up of basic mathematical functions' execution)
* Scikit-Learn              (for metrics functions)
* Pandas                    (for reading-in .csv dataset files)
* Matplotlib                (for the graphs) 
