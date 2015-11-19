# Random forest localization regression using textons

The script trains a random forest on textons of a image and is able to predict
the position of random patches that are either cut out of the image or generated using [draug](https://github.com/tudelft/draug).


## Usage

Main file is `extract_textons_draug.py`.

1. Generate different views from an image using [draug](https://github.com/tudelft/draug).
2. Modify the base_dir in `extract_textons_draug.py`  (in the beginning of the file) to the draug folder 
3. Run `python extract_textons_draug.py` (and make sure that you all all dependencies installed, like numpy, pandas, sklearn, ...)


Basic algorithm:

- Extract 5x5 pixel patches (e.g. 1000) from training image
- Cluster these patches into classes (e.g. 25) using k-means, yielding a texton dictionary
- Either: create random patches (e.g., 10000 patches, one example might be: w x h = 130 x 163 at postion (56, 87))
- Or: Use input patches from [draug](https://github.com/tudelft/draug)
- Extract textons of patches, train a random forest regression, and predict

The code needs quite some refactoring, since it still includes a lot of
experimental stuff.

## Visualization
![Visualization](https://github.com/tudelft/treXton/treXton.png)

