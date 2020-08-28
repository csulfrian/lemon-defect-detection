# Stop! Don't Let That Lemon Pass!
## - or -
## Lemon Defect Detection
### A Capstone Data Science Project
### by Chris Sulfrian

![Lemon](images/lemon.png)

# The Problem
Food waste is a big problem in the US, and using machine learning tools to minimize the chances of food getting wasted would be a very beneficial application of the technology. Consumers tend to be very fickle about the appearance of their produce, so the ideal situation would be to sort every piece with a high degree of scrutiny before it gets distributed to grocery stores food product manufacturers.

This project was designed around the concept of being integrated into an automated sorting/packing line

As of 8/21/2020 wholesale pricing is widely varying, from [$12-$40 per 38lbs](https://www.marketnews.usda.gov/mnp/fv-report-top-filters?startIndex=1&dr=1&rowDisplayMax=25&portal=fv&navClass=&commAbr=LEM&locAbr=&locName=&varName=&region=&commName=LEMONS&navClass&navType=byComm&volume=&type=shipPrice&repType=shipPriceDaily), while nominal retail pricing is [$0.99/lb](https://www.marketnews.usda.gov/mnp/fv-report-retail?portal=fv&category=retail&type=retail&region=NATIONAL&organic=ALL&navClass=FRUITS&commodity=LEMONS). Assuming the mid-range wholesale price, there's a possible margin of $0.25 per pound (33%). As with so much of modern commerce, minimizing waste can mean the difference between staying a viable business and going under.

Each lemon weighs roughly 1/4 pound and represents a potential gross profit of $0.06. Miscategorizing a non-edible lemon has a negative effect in every circumstance, though quantifying that is difficult. Mis-categorizing an industrial lemon to sell at retail represents a potential loss of a sale. 

So our cost/benefit matrix would look something like this:


# The Data

![](images/mosaic_c.png)
*Image credit to the dataset creators*

I found a dataset of 2690 images of lemons, along with annotations for each image that included information about the defects present as well as the overall quality of the image.

I took the initial 9 categories and distilled them down to 3 categories to make decisions about which channel the lemon should be fed through:
- Retail quality
- Industrial quality
- Unsafe for consumption

Requirements for each category:
- Retail (highest quality):
    + image_quality

- Commercial (mid quality):
    + blemish, dark_style_remains, illness

- Non-edible (sent to compost):
    + mould, gangrene


|   id | name               |\n
|-----:|:-------------------|\n
|    1 | image_quality      |\n
|    2 | illness            |\n
|    3 | gangrene           |\n
|    4 | mould              |\n
|    5 | blemish            |\n
|    6 | dark_style_remains |\n
|    7 | artifact           |\n
|    8 | condition          |\n
|    9 | pedicel            |


### The Annotations File
The .JSON formatted annotations file contains multiple entries (roughly 12) per image with segmentation detailing the individual areas of interest on each fruit. 

>`{'id': 6,  
 'iscrowd': 0,  
 'area': 51.0,  
 'category_id': 5,  
 'image_id': 100,  
 'segmentation': [[310.80859375,
   486.7421875,
   308.47572386769025,
   486.20364530558254,
   306.68121739113667,
   488.35705307744684,
   308.11682257237953,
   490.1515595540004,
   310.09077969658756,
   491.40771408758883,
   312.2441874684537,
   491.5871647352433,
   314.7564965356287,
   492.484417973521,
   317.0893549551474,
   491.40771408758883,
   318.70441078404656,
   489.61320761103525,
   318.70441078404656,
   486.921447896204,
   314.7564965356287,
   486.56254660089326,
   312.78253941141884,
   487.81870113447985]],
 'bbox': [306.68121739113667,
  486.20364530558254,
  12.023193392909889,
  6.280772667938436]}`

Which results in an image like this

<img src="images/annotated.jpg" style="width:600px;"/>

# The Technology
- COCO image annotation tools
- Scikit Image
- Scikit Learn
- Amazon EC2 instance


# The Process

The images were scaled down to 128x128 pixels and saved with Scikit Image. I then ran tests on the color images as well as a two transformations of the images. Those included:
- full color
- conversion to grayscale
- Sobel edge detection with both colorspaces

I attempted to run pca on each of the transformation strategies before modeling, but it made the results less accurate in every case.


![](images/first_transforms.jpg)
*Not actual images sent through model - full size for human viewing*

I tested a LogisticRegressionCV model on each of the image transformations on the EC2 instance. 


# The Results

# Future Work
I plan on trying a couple more transformations on the images to put through the logistic regression. After I have a good feeling for which transforms produce the best accuracy, I'd like to expand on the model selection to put the images through. I would like to try:
- Naive Bayes
- a CNN

# References
Dataset: https://github.com/softwaremill/lemon-dataset