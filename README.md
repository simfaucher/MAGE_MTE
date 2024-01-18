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
python -m pip install scikit-image==0.16.2
python -m pip install imutils==0.5.3
conda install pyzmq
python -m pip install pytz==2019.3
```
Install the dependencies for D2Net inside the virtual environment:

```shell
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install h5py imageio imagesize matplotlib numpy scipy tqdm
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

## 3.12 Update

Python 3.12 was release on october 2023 and will receive security updates until october 2028.

We were able to update almost all the dependencies to the latest version. The only one that we couldn't update is [pykson](https://pypi.org/project/pykson/). We tried to update it to the latest version but it was not compatible with the rest of the project. And the package is not maintained anymore. So we decided to keep the old version of pykson that is still working with python 3.12.

We also found out that some packages were is beta version and got released under a new name. So we updated them to the latest version.

### Packages updated version

| Package                                                                          | Old version | New version |
| -------------------------------------------------------------------------------- | ----------- | ----------- |
| [matplotlib](https://pypi.org/project/matplotlib/)                               | 3.1.3       | 3.8.2       |
| [opencv-contrib-python](https://pypi.org/project/opencv-contrib-python/)         | 3.4.2.17    | 4.9.0.80    |
| [pykson](https://pypi.org/project/pykson/)                                       | 0.9.4       | 0.9.4       |
| sklearn(beta) -> [scikit-learn](https://pypi.org/project/scikit-learn/)(release) | 0.0         | 1.3.2       |
| [imutils](https://pypi.org/project/imutils/)                                     | 0.5.3       | 0.5.4       |
| [pyzmq](https://pypi.org/project/pyzmq/)                                         | 18.1.0      | 25.1.2      |
| [numpy](https://pypi.org/project/numpy/)                                         | 1.18.2      | 1.26.3      |
| [pytz](https://pypi.org/project/pytz/)                                           | 2019.3      | 2023.3      |
| [scikit-image](https://pypi.org/project/scikit-image/)                           | 0.16.2      | 0.22.0      |
| [tqdm](https://pypi.org/project/tqdm/)                                           | 4.46.0      | 4.66.1      |
| [h5py](https://pypi.org/project/h5py/)                                           | 2.10.0      | 3.10.0      |
| [imagezmq](https://pypi.org/project/imagezmq/)                                   | 1.0.1       | 1.1.1       |

By updating all these packages, we were able to update the python version from 3.7 to 3.12. We were able to run the communication test successfully with the new version of python.

### Warnings about packages

#### Discontinued packages

While looking for the new version of the packages, we found out that some of them were not maintained anymore (or didn't receive any update since 2022) but were working with the latest version of python. So we decided to keep them. Note that these packages are not used a lot (based on their respective Github stats) so they are most likely never going to be updated or forked.

Here is the list of the packages that are not maintained anymore:

| Package                                        | latest version | last update |
| ---------------------------------------------- | -------------- | ----------- |
| [pykson](https://pypi.org/project/pykson/)     | 0.9.9.8.17     | 26-12-2022  |
| [imutils](https://pypi.org/project/imutils/)   | 0.5.4          | 15-01-2021  |
| [imagezmq](https://pypi.org/project/imagezmq/) | 1.1.1          | 23-05-2020  |

#### Packages included in the repository

We found out that one package was included in the repository because it was not available on PyPI. We decided to remove it from the repository and install it using pip. The package was imagezmq.