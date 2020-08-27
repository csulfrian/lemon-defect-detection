# To run this project

- Open `run_me.py`
- Replace the existing paths with the abolute paths to the raw images & COCO-formatted annotations file on your machine
- Instantiate the `ImagePreprocessor` class & call the `resize_save` method, with `save=True` for the first run. It will take some time while the images are resized, then converted and featurized
- Instantiate the `DatasetBuilder` class & call the `load_data' method