# Python essentials #

# Installation #
There's a whole installation documentation [found here](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine) because installing CNTK might vay depending on your machine, with these factors being relevant:
- Which OS are you using? There's currently support for Windows and Linux (No love for MacOS yet)
- Does you PC have a GPU or only CPU?
- Do you plan to use 1 bit Stochastic Gradient Descent [(1B-SGD)](https://docs.microsoft.com/en-us/cognitive-toolkit/enabling-1bit-sgd)? Scenarios covered here are 1 server with multiple GPUs and/or multiple servers with a single or multiple GPUs.

My current scenario is using Python 3.6  in a Windows laptop with CPU only so I should run this version according to the [installation documentation](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-windows-python?tabs=cntkpy22):

```console
pip install https://cntk.ai/PythonWheel/CPU-Only/cntk-2.2-cp36-cp36m-win_amd64.whl
```
We'll also need to install Anaconda, you can [download it from here](https://www.anaconda.com/download/)  (we'll work with the 4.3.0 version).

## Testing our installation ##
To see if our installation was succesful we can query our CNTK version by running the next line:
```console
python -c "import cntk; print(cntk.__version__)"
```
And the output should be the CNTK version:
```console
2.2
```
## What exactly is Anaconda? ##
