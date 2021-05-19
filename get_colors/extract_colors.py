import glob
import cv2
from PIL import Image, ImageColor
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
import math
import os
import getopt
import sys
from pathlib import Path


def pic_lst(files):
    """ Converts list of file paths into a list of images.
        
        Params: {files: List}
        Returns: List
    """
    photos = []
    for file in files:
        png_image = cv2.imread(file)
        if png_image is None: continue
        image = cv2.cvtColor(png_image, cv2.COLOR_BGR2RGB)
        photos.append(image)
    return photos


# Source: https://realpython.com/python-opencv-color-spaces/
def segmentImage(image):
    """ Returns the segmented version of the input image.
        
        Params: {image: OpenCV InputArray}
        Returns: OpenCV InputArray
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    mask = cv2.inRange(s, 150, 255)
    result = cv2.bitwise_and(image, image, mask=mask)
    indices = np.where(mask==0)
    result[indices[0], indices[1], :] = [255, 255, 255]
    
    return result


def crop_half(img):
    """ Returns a cropped version of the input image with dimensions cut down to half of the original.
        
        Params: {img: OpenCV InputArray}
        Returns: OpenCV InputArray
    """
    print(img.size)
    w, h = img.size
    
    left = w // 4 
    right = 3 * w // 4
    up = h // 4
    down = 3 * h // 4
    crop = img.crop((left, up, right, down))
    return crop


def get_labels(colors, label_colors, labels):
    """ Returns a list of labels that most closely match each color.
        
        Params: {colors: List,
                 label_colors: List,
                 labels: List}
        Returns: List
    """
    out = []
    distances = scipy.spatial.distance.cdist(colors, label_colors)
    for i in range(len(colors)):
        out.append(labels[np.argmin(distances[i])])
    return out


def main():
    
    save = False
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hs")
    except getopt.GetoptError:
        print('Invalid request. Please use the format \'extract_colors.py <-s>\' ')
        sys.exit()
        
    for opt, arg in opts:
        if opt == '-h':
            print('extract_colors.py <-s>')
            sys.exit()
        elif opt == "-s":
            save = True
        else:
            print('Invalid request. Please use the format \'extract_colors.py <-s>\' ')
            sys.exit()
            
    max_photos = 50
    
    # --------- Clear Directories ---------
    print("Clearing directories...")
    [f.unlink() for f in Path(os.path.join(os.curdir, "U-2-Net", "test_data", "u2netp_results", "original_maps")).glob("*") if f.is_file()]
    [f.unlink() for f in Path(os.path.join(os.curdir, "U-2-Net", "test_data", "u2netp_results", "seg_maps")).glob("*") if f.is_file()]
    [f.unlink() for f in Path(os.path.join(os.curdir, "U-2-Net", "test_data", "originals")).glob("*") if f.is_file()]
    [f.unlink() for f in Path(os.path.join(os.curdir, "U-2-Net", "test_data", "segmented")).glob("*") if f.is_file()]
    [f.unlink() for f in Path(os.path.join(os.curdir, "U-2-Net", "test_data", "u2netp_results", "crop_seg_maps")).glob("*") if f.is_file()]
    [f.unlink() for f in Path(os.path.join(os.curdir, "U-2-Net", "test_data", "crop_seg")).glob("*") if f.is_file()]
    [f.unlink() for f in Path(os.path.join(os.curdir, "U-2-Net", "test_data", "cropped")).glob("*") if f.is_file()]
    [f.unlink() for f in Path(os.path.join(os.curdir, "U-2-Net", "test_data", "masked")).glob("*") if f.is_file()]
    [f.unlink() for f in Path(os.path.join(os.curdir, "output_colors")).glob("*") if f.is_file()]
    
    # --------- Loading Images ---------
    all_files = sorted(glob.glob("./input_photos/*.png"))
    all_photos = pic_lst(all_files)
    num_files = len(all_files)
    if num_files > max_photos: save = False
    print(str(num_files) + " images collected")
    rounds = math.ceil(num_files / max_photos)
    os.chdir("U-2-Net")
    
    for i in range(rounds):
        files = all_files[max_photos * i:max_photos * (i + 1)]
        photos = all_photos[max_photos * i:max_photos * (i + 1)]
        print("Processing " + str(len(files)) + " photos...")
        
        
        # --------- Initial Segmentation ---------
        segmented = []
        for num in range(len(photos)):
            name = files[num][files[num].rfind(os.sep) + len(os.sep):]
            or_write = cv2.imwrite(os.path.join(os.curdir, "test_data", "originals") + os.sep + name, 
                      cv2.cvtColor(photos[num], cv2.COLOR_RGB2BGR))
            if not or_write: print("Original photo write error on file: " + files[num])
            seg = segmentImage(photos[num])
            segmented.append(seg)
            seg_write = cv2.imwrite(os.path.join(os.curdir, "test_data", "segmented") + os.sep + name, 
                      cv2.cvtColor(seg, cv2.COLOR_RGB2BGR))
            if not seg_write: print("Segmentation write error on file: " + files[num])
                
        # --------- U-2-Net Object Detection ---------
        # Source: https://github.com/xuebinqin/U-2-Net
        os.system("python u2net_test_modified.py -i " + os.path.join(os.curdir, "originals") + " -o original_maps" + os.sep)
        os.system("python u2net_test_modified.py -i " + os.path.join(os.curdir, "segmented") + " -o seg_maps" + os.sep)
        
        # --------- Apply Masks to Segmented Images ---------    
        or_maps = pic_lst(sorted(glob.glob("./test_data/u2netp_results/original_maps/*.png")))
        seg_maps = pic_lst(sorted(glob.glob("./test_data/u2netp_results/seg_maps/*.png")))  
        
        round2 = []
        for idx in range(len(or_maps)):
            name = files[idx][files[idx].rfind(os.sep) + len(os.sep):]
            thresh = 200
            pic = segmented[idx]
            mask1 = or_maps[idx] < thresh
            mask2 = seg_maps[idx] < thresh
            pic[mask1] = 255
            pic[mask2] = 255

            while (np.average(pic) == 255):
                thresh -= 25
                if thresh < 0:
                    round2.append(files[idx])
                    break
                pic = segmented[idx]
                mask1 = or_maps[idx] < thresh
                mask2 = seg_maps[idx] < thresh
                pic[mask1] = 255
                pic[mask2] = 255
            
            file = cv2.imwrite(os.path.join(os.curdir, "test_data", "masked") + os.sep + name, cv2.cvtColor(pic, cv2.COLOR_RGB2BGR))
            if not file: print("Masked write error on file: ", files[idx])
        
        if not save: 
            [f.unlink() for f in Path(os.path.join(os.curdir, "test_data", "u2netp_results", "original_maps")).glob("*") if f.is_file()]
            [f.unlink() for f in Path(os.path.join(os.curdir, "test_data", "u2netp_results", "seg_maps")).glob("*") if f.is_file()]
            [f.unlink() for f in Path(os.path.join(os.curdir, "test_data", "originals")).glob("*") if f.is_file()]
            [f.unlink() for f in Path(os.path.join(os.curdir, "test_data", "segmented")).glob("*") if f.is_file()]
        
        # --------- Cropping and Re-Processing Blank Images ---------
        if len(round2) > 0:    
            imgs = [Image.open(os.path.join(os.pardir, f[2:])) for f in round2]
            cropped = [crop_half(i) for i in imgs]
            names = [x[x.rfind(os.sep) + len(os.sep):] for x in round2]
            
            for i in range(len(cropped)):
                cropped[i].save(os.path.join(os.curdir, "test_data", "cropped") + os.sep + names[i])
            
            # --------- Initial Segmentation on Cropped Images ---------
            crops = sorted(glob.glob("./test_data/cropped/*.png"))
            crop_pics = pic_lst(crops)
            for i in range(len(crop_pics)):    
                file = cv2.imwrite(os.path.join(os.curdir, "test_data", "crop_seg") + os.sep + names[i],
                                   cv2.cvtColor(segmentImage(crop_pics[i]), cv2.COLOR_RGB2BGR))
                if not file: print("Cropped and segemented image write error on file: ", round2[i])
                    
            # --------- U-2-Net Object Detection on Cropped Images ---------
            # Source: https://github.com/xuebinqin/U-2-Net
            os.system("python u2net_test_modified.py -i " + os.path.join(os.curdir, "crop_seg") + " -o crop_seg_maps" + os.sep)
            
            # --------- Apply Mask to Segmented Images --------- 
            maps_s = pic_lst(sorted(glob.glob("./test_data/u2netp_results/crop_seg_maps/*.png")))
            segmented2 = pic_lst(sorted(glob.glob("./test_data/crop_seg/*.png")))
            
            for idx in range(len(maps_s)):
                thresh = 200
                pic = segmented2[idx]
                mask2 = maps_s[idx] < thresh
                pic[mask2] = 255

                while (np.average(pic) == 255):
                    thresh -= 25
                    if thresh < 0:
                        break
                    pic = segmented2[idx]
                    mask2 = maps_s[idx] < thresh
                    pic[mask2] = 255

                file = cv2.imwrite(os.path.join(os.curdir, "test_data", "masked") + os.sep + names[idx],
                                   cv2.cvtColor(pic, cv2.COLOR_RGB2BGR))
                if not file: print("Cropped and masked image write error on file: ", round2[idx])
        if not save:
            [f.unlink() for f in Path(os.path.join(os.curdir, "test_data", "u2netp_results", "crop_seg_maps")).glob("*") if f.is_file()]
            [f.unlink() for f in Path(os.path.join(os.curdir, "test_data", "crop_seg")).glob("*") if f.is_file()]
            [f.unlink() for f in Path(os.path.join(os.curdir, "test_data", "cropped")).glob("*") if f.is_file()]
        
        # --------- Extracting and Labeling Colors ---------
        print("Labeling colors...")
        final_files = sorted(glob.glob("./test_data/masked/*.png"))
        finals = pic_lst(final_files)
        
        colors = []
        for i in range(len(finals)):
            sums = np.sum(finals[i], axis = 2)
            bool_arr = sums < 255*3

            r_vals = np.extract(bool_arr, finals[i][:, :, 0])
            g_vals = np.extract(bool_arr, finals[i][:, :, 1])
            b_vals = np.extract(bool_arr, finals[i][:, :, 2])

            total_r = np.sum(r_vals)/len(r_vals)
            total_g = np.sum(g_vals)/len(g_vals)
            total_b = np.sum(b_vals)/len(b_vals)

            red = np.full((400, 400), total_r)
            green = np.full((400, 400), total_g)
            blue = np.full((400, 400), total_b)

            result = np.stack((red,green,blue), axis = -1).astype(np.uint8)
            colors.append(result)
            
        rgbs = [(c[:,:,0][0][0], c[:,:,1][0][0], c[:,:,2][0][0]) for c in colors]
        brighter = [tuple(x+25 for x in tup) for tup in rgbs]
        
        labels = ['Mustard Yellow', 'Yellow Brown', 'Brown', 'Dark Green', 'Evergreen', 'Camo Green', 'Light Green',
                  'Red Orange', 'Golden Yellow', 'Red', 'Chalk Grey', 'Dark Brown', 'Greenish Brown', 'Greenish Tan',
                  'Rust Brown', 'Light Brown', 'Orange']
        hex_codes = ['#CEA843', '#B17E1E', '#785A32', '#31361E', '#555E38', '#646421', '#969D36', 
                     '#B35728', '#ffcc00', '#AC2F1C', '#CDCCC8', '#3D3420', '#8F782D', '#9E8825',
                     '#966432', '#916D2D', '#DC8921' ]
        label_colors = [ImageColor.getcolor(hc, "RGB") for hc in hex_codes]
        labeled = get_labels(brighter, label_colors, labels)
        
        for l in range(len(labeled)):
            name = final_files[l][final_files[l].rfind(os.sep) + len(os.sep):]
            result = colors[l]
            file = cv2.imwrite(os.path.join(os.pardir, "output_colors") + os.sep + name[:-4] + "_" + labeled[l] + ".png",
                               cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            if not file: print("Color write error on file: ", files[l])
       
        if not save:
            [f.unlink() for f in Path(os.path.join(os.curdir, "test_data", "masked")).glob("*") if f.is_file()]

                
if __name__ == "__main__":
    main()       