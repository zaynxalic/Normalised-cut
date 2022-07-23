# Image Segmentation via Spectral Clustering with Density Sensitive Similarity Function
![Segmentation results](https://raw.githubusercontent.com/zaynxalic/Normalised-cut/blob/main/result.png?raw=true)
## File structure:
```
├── data/
|   └── cifar-10-python.tar.gz
├── demo.py
├── DSSC.py
├── evaluation.py
├── kmeans.py
├── ncut.py
├── preprocessing.py
├── test0.png
├── test1.png
├── test2.png
├── utils.py
```

# How to run it:
## Reproduce the result in report by:
```
python demo.py 
```
The default label is plane.

You can try different labels by 
```
python demo.py --label='your label'
```

The supported labels in demo are plane, horse and deer.


## Reproduce Segmentation result by three different models
### kmeans:
```
python kmeans.py
```
The default label is using kmeans 3d ++ model.

You can try kmeans 5d ++ model by 
```
python kmeans.py --dim=5
```
### ncut:
```
python ncut.py
```
### DSSC:
```
python DSSC.py
```

# Evaluation:
```
python evaluation.py
```
The model is evaluated by three different metrics -- Accuracy, F-score, NMI. The evaluation evaluates three models' performance on horse, deer and plane. We manually label these pictures as ``test*.png`` for evaluation.

**Note that**: The table below is ran under 10 test pictures, in submission, we only use 3 test images as example.

Algorithm     | ACC          | F-score       | NMI
--------------| -------------|---------------|---------
Kmeans++ 3D   | 0.69         | 0.72          | 0.54
Kmeans++ 5d   | 0.63         | 0.57          | 0.48
Ncut          | 0.71         | 0.66          | 0.52
**DSSC**      | **0.76**     | **0.75**      | **0.62**
