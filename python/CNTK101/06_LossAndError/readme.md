# Usage of Loss and Error #
While CNTK has many loss functions such as **lambda rank**, **cross entropy**, **squared error** and **cosine distance** we'll focus on cross entropy for this entry.
You can check the rest of the loss functions in the [cntk.loss package](https://cntk.ai/pythondocs/cntk.losses.html).

## Cross Entropy with softmax ##
### Softmax 101 ###
*The softmax function is a generalization of the logistic function that **squashes** a K-dimensional vector z of arbitrary real values to a K-dimensional vector σ(z)  of real values in the range [0, 1] that add up to 1*. [Pattern recognition and Machine Learning - Bishop, Christopher(2006)](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)
If you didn't understood that in the first try, don't worry, it's easier to see an example:
```javascript
ourArray = [1,2,3,2,1]
newArray = SoftMax(ourArray)
newArray = [0.111, 0.222, 0.333, 0.222, 0.111]
```

- ourArray is our K-dimensional vector of real values
- SoftMax grabs this values: 1,2,3,2,1 and gives them new values in a way that their new values ranging from 0 to 1 will sum a total of 1
- newArray is our result after applyng the SoftMax function

Indeed, after removing some decimals, if we sum the values of newArray wewill get an approximate result to 1:
```javascript
0.111 + 0.222 + 0.333 + 0.222 + 0.111 = 0.999 (~1)
```
Black magic? Not really, the function basically sums the initial values of our array:
```javascript
total = 1 + 2 + 3 + 2 + 1
total = 9
```
Then divides 1 by the total value:
```javascript
newValue = 1 / total
newValue = 0.111...
```
Now this value is multiplied for each of our original array's values.
```javascript
newArray = [1*newValue, 2*newValue, 3*newValue, 2*newValue, 1*newValue]
newArray = [0.111, 0.222, 0.333, 0.222, 0.111]
``` 
### Cross Entropy ###
Basically, it measures the average number of bits needed to identify an event drawn from a set between two probability distributions *p* and *q*. It's an alternative to the *squared error* function.

## Using cross_entropy_with_softmax ##
The **cross_entropy_with_softmax** function computes the cross entropy between the **target_vector** and the softmax value of the **output_vector**.
The basic implementation of the function receives two parameters: The output_vector and the target_vector:
```python
crossEntropy = cross_entropy_with_softmax(outputLayer, labelShape)
```

Let's recall that the outputLayer and labelShape values come from our previous sample, here's the code so far:
```python
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs
from cntk.ops import input_variable
from cntk.layers import Dense
from cntk.ops import relu

path = "dataset.txt"
featuresShapeValue = 6
labelsShapeValue = 1

featuresShape = input_variable(featuresShapeValue)
labelsShape = input_variable(labelsShapeValue)

ctfdResult = CTFDeserializer(path, StreamDefs(
        features=StreamDef(field='features', shape=featuresShapeValue),
        labels=StreamDef(field='labels', shape=labelsShapeValue)))

reader = MinibatchSource(ctfdResult)

hiddenLayerDimension = 4
hiddenLayerOne = Dense(hiddenLayerDimension, activation=relu)(featuresShape)
outputLayer = Dense(labelsShapeValue, activation=relu)(hiddenLayerOne)

## Cross entropy joins our code so far ##

from cntk.losses import cross_entropy_with_softmax
crossEntropy = cross_entropy_with_softmax(outputLayer, labelsShape)
```
## Classification_Error ##
We'll use the classification_error function for our demo. This and other error functions are located in the [cntk.metrics](https://cntk.ai/pythondocs/cntk.metrics.html) package. It receives the same basic parameters as our previous cross_entropy_with_softmax function: and output_vector and a target_vector:
```python
from cntk.metrics import classification_error
classificationError = classification_error(outputLayer, labelsShape)
```
What this function basically does is to find the index of the highest value in the output_vector and compares it to the actual ground truth label (the index of the hot bit in the target vector)

### So what's a hot bit? ###
To learn what a hot bit is we first need to learn what a hot vector is, and before that we need to learn a little more about hot encoding.

#### Hot encoding ####
Do we have numeric or categorical thinking? When we try to classify 'real world' information we think of categories:
- cats, dogs and cows are animals
- 1, 2 and 3 are numbers
- bike, bycicle and a car are vehicles
- etc...

Some of this categorical data have a natural relationship like our numbers category:
```javascript
1 -> 2 -> 3 -> ... -> infinity 
```
But what about our animals or vehicles category? 
```javascript
cats -> dogs -> cows ? What's even next?
```
We could define a relationship but it's not 'natural' per se. So if we wanted to, say, classify an animal in a picture. How could we define values for our dogs, cats and cows? It would be difficult.
##### Enter numerical data #####
We could turn our labels into numbers using two approaches:
- Integer encoding
- **One-Hot encoding**

Integer encoding basically tells us to asign numbers to our values:
```javascript
bycicle = 1
bike = 2
card = 3
and so on
```
For some variables and algorithms this could be enough. In our vehicles example we could create a hierarchy where the vehicle with most space for passengers gets a higher value.
But what about our animals category? We could try to stablish similar hierarchies like categorize our animals in terms of age, number of legs,etc. But we could complicate ourselves.

And when we complicate ourselves we should think of... **One-Hot encoding** which basically consists of assigning a new binary value for our different values. What does this mean? Let's see our new values:
```javascript
cat = [0 0 1]
dog = [0 1 0]
cow = [1 0 0]
```
So those new values are our **hot vectors** that have a single high (our **hot bit**) and all of the other values are low (0 value).