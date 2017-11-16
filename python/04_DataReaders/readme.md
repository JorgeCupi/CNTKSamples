# Using readers in CNTK #
## CTFDeserializer ##
The **CTFDeserializer** class is located at the [cntk.io]((https://cntk.ai/pythondocs/cntk.io.html)) module and we use it to read text files in a special CNTK format:

```console
|labels 0 1 1 1 0 0 1 |features 1 1 1 0 1 1 0 1
|labels 0 1 0 0 1 0 0 |features 1 0 0 0 1 0 0 0
...

|labels 1 1 0 0 1 0 1 |features 0 0 1 1 0 0 0 1  
```
This format requires a sample per line and there's a pipe '|' delimiter for every name/label in each line.

Our 'dataset' for this example will consist on fake information for hip dysplasia in babies:
```console
labels 1|features 1 1 1 1 0 1
labels 0|features 1 0 1 0 0 0
labels 1|features 0 1 0 0 1 1
labels 1|features 0 1 1 1 0 0
labels 0|features 1 1 1 0 1 1
labels 1|features 1 0 0 1 0 0
labels 0|features 0 1 0 1 0 0
labels 0|features 1 1 0 0 0 0
labels 1|features 0 0 1 0 1 1
labels 0|features 1 1 1 0 1 1
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
labels 1|features 1 1 1 1 0 1
```
Would mean that this baby:
- Did have hip dysplasia
- All mother, father, grandfather and grandmother (mother's side) and grandfather (father's side) had dysplasia when they were babies
- Only the grandmother (from the father's side) didn't have dysplasia when she was a baby

How can we read this dataset? Well:
```python
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs

path = "dataset.txt"
CTFDeserializer(path, StreamDefs(
        features=StreamDef(field='features', shape=6),
        labels=StreamDef(field='labels', shape=1)))
```

What are those lines of code doing?
- The first line of code is just importing some cntk modules, specifically the MinibatchSource, CTFDeserializer, StreamDef and StreamDefs ones 
- The second line just declares a variable 'path' that contains the destination path from our source. (In this case, a file named dataset.txt that is living under the same app.py file)
- The third line is using our CTFDeserializer with two parameters:
    - A path string where we put our path variable 
    - A StreamDefs object that contains:
        - A 'features' parameter where we define:
            - The name of the field within our data set where our 
            - The shape of our features, 6 in our example
        - A 'labels' parameter where we define:
            - The name of the field within our data set where our 
            - The shape of our features, 1 in our example 

