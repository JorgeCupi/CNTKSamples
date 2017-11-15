# Handling variables in CNTK #
We'll learn how to handle and input variables in CNTK in this short demo.

## **input_variable** function ##
Defined in the [cntk.ops.sequence](https://cntk.ai/pythondocs/_modules/cntk/ops/sequence.html) module. the **input_variable** function allows us to create an input in our network so we should use it when we want to define data such as features and labels.

### Usage of **input_variable** ###
It is really simple easy to use it. Say, we want to define a *features* attribute in our CNTK network we would accomplish this by writing:
```python
import cntk
myFeatures = 7
features = cntk.input_variable(myFeatures)
```

Or even shorter:
```python
features = cntk.input_variable(7)
```
### input_variable parameters ###
- shape: Defines the shape of the input variable
- dtype: Defines the type of the variable, its default value is np.float32 and could be set to np.float64
- needs_gradient: Defines whether to back-propagates to it or not
- is_sparse: Defines if the variable is sparse or not
- sequence_axis: Defines a dynamic axis
- name: Defines the name of the function within the network

This *features* result object is a CNTK Variable type (defined in the [cntk.variables](https://cntk.ai/pythondocs/_modules/cntk/variables.html) module) and basically denotes a symbolic entity corresponding to the inputs and outputs of a Function. such Variable-type object can be turned into a Parameter or Constant object (both are defined in the **cntk.variables** module as well).

The **input_variable** function is also able to receive inputs from many sizes thanks to its **shape** attribute. Let's see how to use it.
```python
import cntk
data  = cntk.input_variable(shape=[3,5])
```
This would give us a **data** object with a 3x5 shape (3 rows and 4 columns)

### Usage of **Parameter** ###
What if we needed to define a variable of 3x5 shape but with some data already placed? We would need to use a Parameter class:
```python
import cntk
data = cntk.parameter(shape=(3,5), init=4)
```
What do we have on this data object? Let's see:
```python
data.asarray()

output
array([[ 4.,  4.,  4.,  4.,  4.],
       [ 4.,  4.,  4.,  4.,  4.],
       [ 4.,  4.,  4.,  4.,  4.]], dtype=float32)
```

Note that we could also use:
```python
data.value

# and have the same output result: #
array([[ 4.,  4.,  4.,  4.,  4.],
       [ 4.,  4.,  4.,  4.,  4.],
       [ 4.,  4.,  4.,  4.,  4.]], dtype=float32)
```