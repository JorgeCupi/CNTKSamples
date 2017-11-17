# Testing #
But how good is a model without testing? 

We'll need to define a new path for our test file, a new CTFDeserializer, a new reader and a new input_map
```python
testPath = "test.txt"

ctfdResultTest = CTFDeserializer(testPath, StreamDefs(
        features=StreamDef(field='features', shape=featuresShapeValue),
        labels=StreamDef(field='labels', shape=labelsShapeValue)))

readerTest = MinibatchSource(ctfdResultTest)

inputMapTest = = {
    featuresShape: readerTest.streams.features,
    labelsShape: readerTest.streams.labels
}
```

Remember the **next_minibatch** method from our MinibatchSource class?For testing purposes we'll leverage it along another method:
- We'll use the **next_minibatch**  method to grab a number of samples within our Minibatch from our **readerTest** object
- We'll use the **test_minibatch** object to feed this Minibatch to our **trainer** object

Let's implement it:

```python
minibatchSizeTest = 25
numberOfSamplesTest = 312
minibatchesToTest = numberOfSamplesTest / minibatchSizeTest
testResult = 0.0
for i in range(0, int(minibatchesToTest)):
    mb = readerTest.next_minibatch(minibatch_size_in_samples = minibatchSizeTest, input_map=inputMapTest)
    evalError = trainer.test_minibatch(mb)
    testResult = testResult + evalError

averageClassificationError = testResult / minibatchesToTest
```

If we print our **averageClassificationError** value we'll get an approx result of 0.33012. Which means that's our model error. 
What does this mean? Means that if we introduce new data to determine a baby will have hip dysplasia or not, there's only a 67% chance that our model will give the right result.

Well, that sucks. A model with only a 67% accuracy won't make it to the production. But let's remember that we're using fake generated data so it is understandable. To decrease this error there are lots of things we could do:
- Choose a different algorithm
- Change the number of hidden layers / nodes
- Improve the dataset (use real values!)
- Define if we're better with one-hot encoding or not
- A big ETC...

That's it for this CNTK introduction mini series! Hope you enjoyed it. 

On future posts we'll cover more specific scenarios surrounding better algorithms, error functions, data readers, etc, with what've learned this time.