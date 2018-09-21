### pygunrock

Simple python implementation of [gunrock](https://github.com/gunrock/gunrock/) APIs

#### Quickstart

```
conda create -n pygunrock_env python=3.6 pip -y
source activate pygunrock_env
pip install -r requirements.txt

python apps/hello.py        # Compute degrees
python apps/sssp.py --src 1 # Compute SSSP
```

#### About

[gunrock](https://github.com/gunrock/gunrock) is a CUDA library for graph-processing designed specifically for the GPU.  It is highly performant, but [apps](https://github.com/gunrock/gunrock/tree/dev-refactor/tests) must be written in C++/CUDA, which is not always the easiest way to prototype a new algorithm.  

`pygunrock` is a pure Python implementation of the `gunrock` API, optimized for rapid prototyping and ease of understanding.

`pygunrock` is _not_ a Python API for gunrock.  It's intended to be used as a development environment, where the programmer can map their algorithm to the `gunrock` abstractions, test for correctness, and then port the `pygunrock` Python code to `gunrock` C++ code.  When developing complex apps, `pygunrock` can serve as a useful tool for accelerating programmer productivity.

#### Usage

We have a number of apps implemented in both `pygunrock` and `gunrock`
 - `pygunrock`: https://github.com/bkj/pygunrock/tree/master/apps
 - `gunrock`: 
  - https://github.com/gunrock/gunrock/tree/dev-refactor/tests
  - https://github.com/gunrock/gunrock/tree/dev-refactor/gunrock/app

For example, SSSP is available in [pygunrock](https://github.com/bkj/pygunrock/blob/master/apps/sssp.py) and in `gunrock`, in the [tests/sssp](https://github.com/gunrock/gunrock/tree/dev-refactor/tests/sssp) and [gunrock/app/sssp/](https://github.com/gunrock/gunrock/tree/dev-refactor/gunrock/app/sssp) directories.  A side-by-side comparison of the implementations shows their similarity.  For instance, compare:
 - https://github.com/gunrock/gunrock/blob/dev-refactor/gunrock/app/sssp/sssp_enactor.cuh#L88
 - https://github.com/bkj/pygunrock/blob/master/apps/sssp.py#L56
 
or

 - https://github.com/gunrock/gunrock/blob/dev-refactor/gunrock/app/sssp/sssp_app.cu#L97
 - https://github.com/bkj/pygunrock/blob/master/apps/sssp.py#L101
 
