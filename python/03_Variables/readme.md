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

### Variables of different sizes ###
The **input_variable** function is able to receive inputs from many sizes thanks to its **shape** attribute. Let's see how to use it.