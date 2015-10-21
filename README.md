# Random forest localization regression using textons

The script trains a random forest on textons of a image and is able to predict
the position of random patches that are cut out of the image. 

Main file is `extract_textons.py`.

Basic algorithm:


Training:

- Extract 5x5 pixel patches (e.g. 1000) from training image
- Cluster these patches into classes (e.g. 25) using k-means, yielding a texton dictionary
- Create random patches (e.g., 10000 patches, one example might be: w x h = 130 x 163 at postion (56, 87))




The code needs refactoring, since it still includes a lot of
experimental stuff.
