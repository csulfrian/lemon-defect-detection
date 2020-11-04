### To run the Convolutional Neural Network - Xception model

- Pre-classify your images, placing each class into its own named folder
- Open `xception.py`
- Scroll to the `if __name__ == '__main__':` block
- Set the variable `direc` to the path containing your classified images.
- Save and close the file


- Run `python src/xception.py` from the home folder of the project. 

### To run the Logistic Ression part of this project:

- Open `model.py`
- Replace the existing paths with the paths to the raw images & COCO-formatted annotations file on your machine.
- Instantiate the `ImagePreprocessor` class & call the `resize_save` method, with `save=True` & `color=True` for the first run. It will take some time while the images are resized, then converted and featurized. Subsequent runs will be much faster.
- Instantiate the `DatasetBuilder` class & assign `X, y` to `load_data()`
- Do your Train/Test split
- Switch out the model you desire if you desire
- Choose your chosen metrics


- Run `python src/model.py` from the home folder of the project.
