###
#
#       @Brief          run.py
#       @Details        Testing code for EventPointNet : Release Version
#       @Org            Robot Learning Lab(https://rllab.snu.ac.kr), Seoul National University
#       @Author         Howoong Jun (howoong.jun@rllab.snu.ac.kr)
#       @Date           Aug. 12, 2021
#       @Version        v0.1
#
###

from EventPointNet.eventpointnet import CEventPointNet
import argparse, time
from skimage import io, color
from skimage.transform import resize
import numpy as np
from lcore.hal import eSettingCmd

parser = argparse.ArgumentParser(description='Test EventPointNet')
parser.add_argument('--width', '-W', type=int, default=None, dest='width',
                    help='Width for resize image')
parser.add_argument('--height', '-H', type=int, default=None, dest='height',
                    help='Height for resize image')
args = parser.parse_args()

if __name__ == "__main__":
    # Read Image
    oImage = io.imread("./test.jpg")
    if(args.width is not None and args.height is not None):
        oImage = resize(oImage, (args.height, args.width))
        oImage = (oImage * 255).astype(np.uint8)
    # Convert RGB to Grayscale and rescale pixel value into [0, 255]
    oImageGray = (color.rgb2gray(oImage) * 255).astype(np.uint8)
    oImageGray = np.expand_dims(np.asarray(oImageGray), axis=0)
    
    # EventPointNet object
    oEventPointNet = CEventPointNet()
    oEventPointNet.Open()
    oEventPointNet.Setting(eSettingCmd.eSettingCmd_IMAGE_DATA_GRAY, oImageGray)
    ckTime = time.time()
    vKpt, vDesc, vHeatmap = oEventPointNet.Read()