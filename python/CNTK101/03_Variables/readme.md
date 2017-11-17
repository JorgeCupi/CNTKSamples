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

## Usage of **Parameter** ##
What if we needed to define a variable of 3x5 shape but with some data already placed? We would need to use a Parameter class:
```python
import cntk
data = cntk.parameter(shape=(3,5),init=4)
# we could also define our data object by this: #
data = cntk.parameter((3,5),init = 4)
# or #
data = cntk.parameter((3,5),4)
```
What do we have on this data object? Let's see:
```python
data.asarray()
# Note that we could also use: #  
data.value

# and have the same results:
output
array([[ 4.,  4.,  4.,  4.,  4.],
       [ 4.,  4.,  4.,  4.,  4.],
       [ 4.,  4.,  4.,  4.,  4.]], dtype=float32)
```
## Usage of **Constant** ##
Other than our now typical **Variable** and **Parameter** classes we also have a **Constant** class defined in our [cntk.variables](https://cntk.ai/pythondocs/_modules/cntk/variables.html) module:
```python
import cntk
data = cntk.Constant(6,shape = (3,4))
# we could also define our data object by this: #
data = cntk.parameter(6,(3,4))

#output#
data.value # data.asarray() also works#

array([[ 6.,  6.,  6.,  6.],
       [ 6.,  6.,  6.,  6.],
       [ 6.,  6.,  6.,  6.]], dtype=float32)
```

## Parameter vs Constant ##
By the examples above it looks like both classe are almost identical. Let's compare them:
- Both are Variables so they inherit all of their methods
- Both can be scalar, vector, matrix or tensor of floating point numbers
- A Parameter object is trainable, a Constant is not
- A Parameter object can be modified by a training procedure, a Constant cannot be modified

## Usage of Record ##
A record is an inmutable singleton class that holds keyword arguments:
```python
import cntk.variables as var
record = var.Record(x = 23, y = 32, z = 55)

# Printing the record's values #
record.x
record.y
record.z
```

### Updating a record value ###
We just use the **updated_with** method indicating which value we want to update:
```python
import cntk.variables as var
record = var.Record(x = 23, y = 32, z = 55)
record = record.updated_with(x = 42)
```