### To run the Logistic Ression part of this project:

- Open `model.py`
- Replace the existing paths with the paths to the raw images & COCO-formatted annotations file on your machine.
- Instantiate the `ImagePreprocessor` class & call the `resize_save` method, with `save=True` & `color=True` for the first run. It will take some time while the images are resized, then converted and featurized. Subsequent runs will be much faster.
- Instantiate the `DatasetBuilder` class & assign `X, y` to `load_data()`
- Do your Train/Test split
- Switch out the model you desire if you desire
- Choose your chosen metrics


- Run `model.py` from the home folder of the project.