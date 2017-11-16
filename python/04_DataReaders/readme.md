# Using readers in CNTK #
## CTFDeserializer ##
The **CTFDeserializer** class is located at the [cntk.io]((https://cntk.ai/pythondocs/cntk.io.html)) module and we use it to read text files in a special CNTK format (CNTK Text format = CTF):

```console
|labels 0 1 1 1 0 0 1 |features 1 1 1 0 1 1 0 1
|labels 0 1 0 0 1 0 0 |features 1 0 0 0 1 0 0 0
...

|labels 1 1 0 0 1 0 1 |features 0 0 1 1 0 0 0 1  

```
>NOTE: It is really IMPORTANT to note that in your dataset file you have to let a blank line at the end. As an example, if your data set has 100000 entries, your file should have 100001 lines.

This format requires a sample per line and there's a pipe '|' delimiter for every name/label in each line.

Our 'dataset' for this example will consist on fake information for hip dysplasia in babies:
```console
|labels 1 |features 1 1 1 1 0 0
|labels 0 |features 1 0 1 0 0 0
|labels 1 |features 0 1 0 0 1 1
|labels 1 |features 0 1 1 1 0 0
|labels 1 |features 1 1 1 0 1 1
|labels 0 |features 1 0 0 1 0 0
|labels 0 |features 0 1 0 1 0 0
|labels 0 |features 1 1 0 0 0 0
|labels 1 |features 0 0 1 0 1 1
|labels 1 |features 1 1 1 0 1 1
|labels 1 |features 1 1 1 1 0 0
|labels 1 |features 1 0 1 0 0 0

```

With the 'labels' attribute being our truly labels representing:
- 1 for baby being born with hip dysplasia
- 0 for baby being born without hip dysplasia

And our 'features' would be:
- 1st column: Mother of the baby had dysplasia when she was a baby
- 2nd column: Grandmother (Mother's side) of the baby had dysplasia when she was a baby
- 3rd column: Grandfather (Mother's side) of the baby had dysplasia when she was a baby
- 4th column: Father of the baby had dysplasia when she was a baby
- 5th column: Grandmother (Father's side) of the baby had dysplasia when she was a baby
- 6th column: Grandfather (Father's side) of the baby had dysplasia when she was a baby

So this line:
```console
|labels 1 |features 1 1 1 1 0 0
```
Would mean that:
- This baby did have hip dysplasia
- The mother, father, grandfather and grandmother (mother's side) and had dysplasia when they were babies
- The grandmother and grandfather (from the father's side) didn't have dysplasia when they were babies

How can we read this dataset? Well:
```python
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs

path = "dataset.txt"
ctfdResult = CTFDeserializer(path, StreamDefs(
        features=StreamDef(field='features', shape=6),
        labels=StreamDef(field='labels', shape=1)))
```

What are those lines of code doing?
- The first line of code is just importing some cntk modules, specifically the MinibatchSource, CTFDeserializer, StreamDef and StreamDefs ones 
- The second line just declares a variable 'path' that contains the filename from our data source. (In this case, a file named dataset.txt that is living under the same app.py file)
- The third line is using our CTFDeserializer with two parameters:
    - A path string where we put our path variable 
    - A StreamDefs object (that works like a dictionary) and contains StreamDef objects:
        - A 'features' StreamDef object where we define:
            - The name of the field within our data set where our 
            - The shape of our features, 6 in our example
        - A 'labels' StreamDef object where we define:
            - The name of the field within our data set where our 
            - The shape of our features, 1 in our example 

Now we have to dump that CTFDeserializer method result into an object, and for that we'll use an object called MinibatchSource:
```python
reader = MinibatchSource(ctfdResult)
```
We'll use this reader in a future when we'll start to train our models. If you want to learn more MinibatchSources

### Reading the dataset ###
Now that we have our dataset loaded into a MinibatchSource object called **reader** we can access that dataset  using the **next_minibatch** method. Among other parameters, this method receives an int parameter called **minibatch_size_in_samples**
defining the number of rows we want to acquire:

```python
data = reader.next_minibatch(1)
```
>[There's this document]() that explains quite well the logic behind the **minibatch_size_in_samples** parameter. Simply put:  It denotes the number of samples **between model updates**. So it's not a sequential acquisition of samples but an acquisition between model updates which will vary for each epoch. (Seriously, read the post and you'll thank me) 

To see the content of that MinibatchData object we need to indicate which field we are looking at. Let's remember how our dataset looks like right now:

```console
|labels 1 |features 1 1 1 1 0 0
|labels 0 |features 1 0 1 0 0 0
|labels 1 |features 0 1 0 0 1 1
|labels 1 |features 0 1 1 1 0 0
|labels 1 |features 1 1 1 0 1 1
|labels 0 |features 1 0 0 1 0 0
|labels 0 |features 0 1 0 1 0 0
|labels 0 |features 1 1 0 0 0 0
|labels 1 |features 0 0 1 0 1 1
|labels 1 |features 1 1 1 0 1 1
|labels 1 |features 1 1 1 1 0 0
|labels 1 |features 1 0 1 0 0 0

```
So, if we wanted to check the **features** value we would have to write something like this:
```python
featuresData = data[reader.streams['features']]
featuresData.asarray()
labelsData = data[reader.streams['labels']]
labelsData.asarray()
```