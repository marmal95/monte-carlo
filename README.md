# Monte Carlo

The project contains sequential and different parallel implementations of **Monte Carlo - π approximation** algorithm.<br />
The project was created for the needs of master's thesis of Computer Science studies.<br/>
The aim of the project was to compare performance of sequential and parallel implementations, depending on threads and processes number used for computing, with the use of different approaches and technologies.

## Functionality
π value is approximated using [Monte Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method).
Each implementation contains **POINTS** constant which determines a number of random points to be generated to approximate π value.

## Technologies

* C++
* [OpenMP](https://en.wikipedia.org/wiki/OpenMP)
* [OpenMPI](https://en.wikipedia.org/wiki/Open_MPI)
* [CUDA](https://en.wikipedia.org/wiki/CUDA)
    * [Thrust](https://github.com/NVIDIA/thrust)

## Implementations

The project contains a few **Monte Carlo method** implementations:
* Sequential implementation - using pure C++
* Multithreaded implementation with the use of **OpenMP**
* Parallel implementation with the use of **OpenMPI**
* Massive parallel implementation with use of **CUDA**

## Installation

### Dependencies
* Visual Studio (C++17 support)
* OpenMP support enabled in Visual Studio (for MonteCarlo-OpenMP project)
* OpenMPI implementation installed e.g. Microsoft MPI
* CUDA installed (+ CUDA capable graphic card)
* Thrust

### Repository

```sh
$ git clone https://github.com/marmal95/monte-carlo.git
```

### Build

The Visual Studio solution contains four projects inside - responding four implementations mentioned in [Implementations](#Implementations) section.
<br/>
Build whole solution by choosing:
```
Build > Build Solution
```
from top menu, or right-click specific project in **Solution Explorer** and choose:
```
Build
```

### Run
Right-click on chosen project in **Solution Explorer** view and click **Set as Startup Project**.<br/>
Click **F5** or choose **Debug >> Start Debugging** from top menu. 


## Customization

### OpenMP

Preferred number of threads in OpenMP implementation used for computing may be changed with function call:
```
omp_set_num_threads(NUM_THREADS)
```
which is called at the beginning of **main()** function.


### OpenMPI

Preferred number of processes is passed as parameter to **mpiexec** command.<br>
The value may be changed in **Visual Studio** in: **Project > Properties > Configuration Properties > Debugging**.<br>
```
Command             mpiexec.exe
Command Arguments   -n 4 "$(TargetPath)"
```


### CUDA

Preferred size of grid used for computing is specified inside **Algorithm.cu** file and can be modified with the following constants:<br>
```
const std::size_t POINTS = 10'000'000'000;
const std::size_t THREADS_PER_BLOCK = 32;
const std::size_t NUM_BLOCKS = 640;
```
