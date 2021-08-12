###
#
#       @Brief          hal.py
#       @Details        Visual Localization core abstract class
#       @Org            Robot Learning Lab(https://rllab.snu.ac.kr), Seoul National University
#       @Author         Howoong Jun (howoong.jun@rllab.snu.ac.kr)
#       @Date           Mar. 03, 2021
#       @Version        v0.7
#
###

from abc import *
from enum import IntEnum

class eSettingCmd(IntEnum):
    eSettingCmd_NONE = 1
    eSettingCmd_IMAGE_DATA_GRAY = 2
    eSettingCmd_IMAGE_DATA = 3
    eSettingCmd_IMAGE_CHANNEL = 4
    eSettingCmd_CONFIG = 5
    eSettingCmd_THRESHOLD = 6
    
class CVisualLocalizationCore(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        print("Visual Localization Core Constructor")
        
    @abstractmethod
    def __del__(self):
        print("Visual Localization Core Destructor")

    @abstractmethod
    def Open(self):
        print("Visual Localization Core Open")
    
    @abstractmethod
    def Close(self):
        print("Visual Localization Core Close")
    
    @abstractmethod
    def Write(self):
        print("Visual Localization Core Write")

    @abstractmethod
    def Read(self):
        print("Visual Localization Core Read")

    @abstractmethod
    def Setting(self):
        print("Visual Localization Core Control")

    @abstractmethod
    def Reset(self):
        print("Visual Localization Core Reset")
        
