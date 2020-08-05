# ACDMotionTrackingEngine

Motion tracking engine for the project UTC ACD, using computer vision to control plane engines. It is used by the ACD client as a server to validate images.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for testing purposes only.


### Prerequisites

The project runs with Python 3.6+ (tested on 3.7.3) and Python libraries packaged using Anaconda. Any desktop operating system supported by Anaconda should work.

### Installing

A regular installation of Python and the project's dependencies would work, but it is preferable to use Anaconda to create a separate virtual environment with the good version of the libraries. This can prevent this installation to interfere with possible others.

First, install Anaconda following the instructions from its [website](https://www.anaconda.com/distribution/ "Anaconda website"). Once installed, open Anaconda prompt on Windows and on Unix systems, open a terminal.

Create the virtual envionment `mte-env` for the project:

```shell
conda create -n mte-env python=3.7.3
```

Then, activate the environment:

```shell
conda activate mte-env
```

Finally, install the dependencies inside the virtual environment:

```shell
python -m pip install matplotlib==3.1.3
python -m pip install opencv-contrib-python==3.4.2.17
python -m pip install pykson==0.9.4
python -m pip install sklearn==0.0
python -m pip install imutils==0.5.3
conda install pyzmq
python -m pip install pytz==2019.3
```
Install the dependencies for D2Net inside the virtual environment:

```shell
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install h5py imageio imagesize matplotlib numpy scipy tqdm
conda install scikit-image
```

## Running the programs

### Launch

The activating step should be executed each time you wish to run the project:

```shell
conda activate mte-env
```

You should see `(mte-env)` at the beginning of the current shell line.

Navigate to the folder where the programs are and run the server:
```shell
python Server.py
```

You can see how to parametrate the server by launching instead:
```shell
python Server.py --help
```

Run the client in another terminal with the Anaconda environment `(mte-env)` activated:
```shell
python Client.py
```

### Usage of the programs

The server is by default in prelearning mode. Press the 2 key with the `Debug` window activated to learn the current image of the video stream.
The learning mode is printed in the server command window and then the server goes back to prelearning mode.

You can then press the 3 key on the `Debug` window to launch the recognition mode. The `Transformed` window should appear. When the recognition works, the image is warped to match the learning image and printed in the `Transformed` window. When it fails, the resized image of the video stream is showed. The state of the recognition is also printed in the client command window.

To quit the programs, first press the Q key on the `Debug` window and then press CTRL+C on the server command window.

### Deactivation

Note: If you want to deactivate the environment to go back to your regular shell, just type:

```shell
conda deactivate
```

## Usage of Docker

### Installing

To use the MTE in its Docker environment, first install Docker following the instructions on its [website](https://www.docker.com/get-started/ "Docker website").

Be sure to be at the root of the project and build the image :

```shell
docker build .
```

The output should print the image ID at the end of the build, something like ```Successfully built <imageID>```.

### Running

You can run the Motion tracking server by entering the following command and replacing ```imageID``` by the actual image ID you had at the installation:

```shell
docker run -p 5555:5555 <imageID>
```

If you lost the image ID, you can find it running:
```shell
docker image list
```