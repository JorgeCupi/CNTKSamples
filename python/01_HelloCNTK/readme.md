# Hello CNTK #
For our first CNTK samples we'll learn how to do basic arithmetics on tensors and while we're at it we'll learn more CNTK's elementals.

If we haven't install CNTK in our root environment we'll have to import it in our files:
```python
import cntk
```
>TIP: To check if CNTK is in your root invironment just type **cntk.__version__** once you're running python. If it is not installed you'll have to run **import cntk** first.

The most basic linear operations are defined in the [cntk.ops module](https://cntk.ai/pythondocs/_modules/cntk/ops.html):
- Addition
- Substraction
- Multiplication
- Division
- Exponent power

Note that there are also more complex operations defined such as:
- Convolution operations
- Tensor comparison operations
- Reshaping operations
- Normalization operations
- Others

On this post, we'll look only at some of the linear operations. 
## Syntaxis ##
Most of them have a similar arguments pattern:
```console
operationName(arg1, arg2, .., argN, functionName)
```
- arg1: The first tensor
- arg2: The second tensor
- argN: The N-th tensor
- functionName: Optional, a name for the current operation

## Operations ##
Let's define two first tensors:
```python
A = [1,2,3]
B = [4,5,6]
```
#### Adding two tensors ####
```python
cntk.plus(A,B).eval()

output:
[ 5.  7.  9.]
```

#### Substracting two tensors ####
```python
cntk.minus(A,B).eval()

output:
[-3. -3. -3.]
```

#### Multiplying two tensors ####
```python
cntk.times(A,B).eval()

output
[[  4.   5.   6.]
 [ 12.  15.  18.]
 [ 16.  20.  24.]]
```

#### Dividing two tensors ####
```python
cntk.element_divide([4,32, 15], [2, 4, 5]).eval()

output
[ 2.  8.  3.]
```

#### Exponential power to a tensor's values: ####
```python
cntk.pow(A,B).eval()

output
[  1.   9.  64.]
```

#### Defining the minimum or maximum single values of multiple tensors ####
```python
cntk.element_min(A,B, [2,1,0]).eval()
output 
[ 1.  1.  0.]


cntk.element_max(A,B,[2,9,0]).eval()

output
[ 4.  9.  6.]
```