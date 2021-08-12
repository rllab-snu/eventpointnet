###
#
#       @Brief          eventpointnet.py
#       @Details        EventPointNet model main class: Release Version
#       @Org            Robot Learning Lab(https://rllab.snu.ac.kr), Seoul National University
#       @Author         Howoong Jun (howoong.jun@rllab.snu.ac.kr)
#       @Date           Aug. 12, 2021
#       @Version        v0.1
#
###


from lcore.hal import *
import EventPointNet.nets as nets
import numpy as np
import torch, cv2
from common.Log import DebugPrint
import time

class CEventPointNet(CVisualLocalizationCore):
    def __init__(self):
        self.softmax = torch.nn.Softmax2d()
        self.__bDescSift = True

    def __del__(self):
        print("CEventPointNet Destructor!")

    def Open(self):
        self.__gpuCheck = torch.cuda.is_available()
        self.__device = "cuda" if self.__gpuCheck else "cpu"
        self.__oQueryModel = nets.CEventPointNet().to(self.__device)
        if(self.__gpuCheck):
            self.__oQueryModel.load_state_dict(torch.load("./EventPointNet/checkpoints/checkpoint_detector.pth"))
            DebugPrint().info("Using GPU..")
        else:
            self.__oQueryModel.load_state_dict(torch.load("./EventPointNet/checkpoints/checkpoint_detector.pth", map_location=torch.device("cpu")))
            DebugPrint().info("Using CPU..")
        if(self.__bDescSift == True):
            DebugPrint().info("Descriptor: SIFT")
            self.__oSift = cv2.SIFT_create()
            
        DebugPrint().info("Load Model Completed!")

    def Close(self):
        print("CEventPointNet Close!")

    def Write(self):
        print("CEventPointNet Write!")
        
    def Read(self):
        with torch.no_grad():
            self.__oQueryModel.eval()
            kptDist = self.__oQueryModel.forward(self.__Image)
            kptDist = self.softmax(kptDist)
            kptDist = torch.exp(kptDist)
            kptDist = torch.div(kptDist, (torch.sum(kptDist[0], axis=0)+.00001))
            kptDist = kptDist[:,:-1,:]
            kptDist = torch.nn.functional.pixel_shuffle(kptDist, 8)
            kptDist = kptDist.data.cpu().numpy()
            kpt, desc, heatmap = self.__GenerateLocalFeature(kptDist, None)
            return kpt, desc, heatmap

    def Setting(self, eCommand:int, Value=None):
        SetCmd = eSettingCmd(eCommand)

        if(SetCmd == eSettingCmd.eSettingCmd_IMAGE_DATA_GRAY):
            self.__ImageOriginal = np.asarray(Value)
            self.__Image = np.expand_dims(np.asarray(Value), axis=1)
            self.__Image = torch.from_numpy(self.__Image).to(self.__device, dtype=torch.float)

    def Reset(self):
        self.__Image = None

    def __GenerateLocalFeature(self, keypoint_distribution, descriptor_distribution):
        heatmap = np.squeeze(keypoint_distribution, axis=0)
        heatmap = np.squeeze(heatmap, axis=0)
        heatmap_aligned = heatmap.reshape(-1)
        heatmap_aligned = np.sort(heatmap_aligned)[::-1]
        xs, ys = np.where(heatmap >= 0.015387)
        vKpt = []
        vDesc = []
        H, W = heatmap.shape
        pts = np.zeros((3, len(xs)))
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = self.__Nms_fast(pts, H, W, 5)
        ys = pts[0, :]
        xs = pts[1, :]
        if(len(self.__ImageOriginal.shape) >= 3):
            self.__ImageOriginal = np.squeeze(self.__ImageOriginal, axis=0)
        
        targetImg = self.__Image
        targetImg = torch.squeeze(torch.squeeze(targetImg, axis=0), axis=0)
        uHeight, uWidth = targetImg.shape
        uOffset = 5
        for kptNo in range(len(xs)):
            if(xs[kptNo] > uHeight - uOffset or xs[kptNo] < uOffset or ys[kptNo] > uWidth - uOffset or ys[kptNo] < uOffset): continue
            if(not self.__bDescSift):
                desc = descriptor_distribution[0][:, int(xs[kptNo]), int(ys[kptNo])]
                vDesc.append(desc)
            vKpt_tmp = cv2.KeyPoint(int(ys[kptNo]), int(xs[kptNo]), 5.0)
            vKpt.append(vKpt_tmp)
        if(self.__bDescSift): _, vDesc = self.__oSift.compute(self.__ImageOriginal, vKpt)
        vDesc = np.array(vDesc)

        oHeatmap = ((heatmap - np.min(heatmap)) * 255 / (np.max(heatmap) - np.min(heatmap))).astype(np.uint8)
        return vKpt, vDesc, oHeatmap

    def __Nms_fast(self, in_corners, H, W, dist_thresh):
        mGrid = np.zeros((H, W)).astype(int) 
        mInds = np.zeros((H, W)).astype(int) 
        
        inds1 = np.argsort(-in_corners[2,:])
        corners = in_corners[:,inds1]
        rcorners = corners[:2,:].round().astype(int) 
        
        if rcorners.shape[1] == 0:
            return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
            return out, np.zeros((1)).astype(int)
        
        for i, rc in enumerate(rcorners.T):
            mGrid[rcorners[1,i], rcorners[0,i]] = 1
            mInds[rcorners[1,i], rcorners[0,i]] = i
        
        pad = dist_thresh
        mGrid = np.pad(mGrid, ((pad,pad), (pad,pad)), mode='constant')
        
        count = 0
        for i, rc in enumerate(rcorners.T):
        
            pt = (rc[0]+pad, rc[1]+pad)
            if mGrid[pt[1], pt[0]] == 1:
                mGrid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
                mGrid[pt[1], pt[0]] = -1
                count += 1
        
        keepy, keepx = np.where(mGrid==-1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = mInds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds