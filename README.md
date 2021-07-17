# colorRecognition

## Installation
To avoid errors, make sure you have a 64-bit Python installation. To find out which version you have, enter `python` on the command line.
Once you’ve confirmed your Python version, run

`pip install -r requirements.txt`

## Running the program
Place the original stool images in the `input_photos` folder and run 

`python extract_colors.py`

### Options
You can add an optional `-s` flag at the end if you’d like to save the intermediate files such as segmented photos, cropped versions, and saliency maps.

`python extract_colors.py -s`

The -s option is only functional if there are 50 or less photos in the `input_photos` directory. Inputs of greater than 50 photos are processed in batches, so intermediary files are automatically deleted.
The intermediary files can be found in the folder `get_photos/U-2-Net/test_data`. Please reference the Spring 2021 report for further documentation on the contents of these folders and the overall color recognition algorithm. 

## Credits
Part of this code is taken from https://github.com/xuebinqin/U-2-Net 
