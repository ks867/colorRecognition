import numpy as np
import cv2


def color_avg(photos, name):
    """ Saves the average color of a list of color swatches as a new color swatch with the given name.
        
        Params: {photos: List,
                 name: String}
        Returns: None
    """
    rgbs = [np.array((c[:,:,0][0][0], c[:,:,1][0][0], c[:,:,2][0][0])) for c in photos]
    brighter = [x+25 for x in rgbs]
    avg = np.sum(brighter, axis = 0)/len(brighter)
    print(brighter)
    print(np.sum(brighter, axis = 0))
    print(len(brighter))
    print(avg)
    
    red = np.full((400, 400), avg[0])
    green = np.full((400, 400), avg[1])
    blue = np.full((400, 400), avg[2])

    result = np.stack((red,green,blue), axis = -1).astype(np.uint8)
    file = cv2.imwrite('./' + name + ".png", 
                       cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    if not file: print("not", name)
        

def write_rgb_colors(labels, label_colors, folder):
    """ Creates label swatches and saves them to the input folder.
        
        Params: {labels: List,
                 label_colors: List,
                 folder: List}
        Returns: List
    """
    for l in range(len(labels)):
        if label_colors[l][0] < 25 or label_colors[l][1] < 25 or label_colors[l][2] < 25: diff = 0
        else: diff = 25

        red = np.full((400, 400), label_colors[l][0] - diff)
        green = np.full((400, 400), label_colors[l][1] - diff)
        blue = np.full((400, 400), label_colors[l][2] - diff)

        result = np.stack((red,green,blue), axis = -1).astype(np.uint8)
        file = cv2.imwrite(folder +'/' + labels[l] + ".png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        if not file: print("not", color_files[l][:-4])
            
 