# from .vimeo90k import Vimeo90KDataset, VideoTestVimeo90KDataset
from .mfqev2 import MFQEv2Dataset, VideoTestMFQEv2Dataset,MFQEv2HQDataset,VideoTestMFQEv2HQDataset,MFQEv2BetaDataset,MFQEv2RTDataset,VideoTestMFQEv2RTDataset
# from .ntire21 import NTIRE21Dataset, VideoTestNTIRE21Dataset,NTIRE21ValDataset
# from .ntire21v2 import NTIRE21v2Dataset
# from .ntire21v3 import NTIRE21v3Dataset,NTIRE21v3ValDataset,VideoTestHQR4NTIRE21Dataset
# from .concatdataset import ConcatDataset
# from .ntire21v3_3 import NTIRE21v3R3Dataset,NTIRE21v3R3ValDataset
# from .mfqergb import MFQERGB,MFQERGBTestDataset
__all__ = [
    'Vimeo90KDataset', 'VideoTestVimeo90KDataset', 
    'MFQEv2Dataset', 'VideoTestMFQEv2Dataset', 
    'NTIRE21Dataset', 'VideoTestNTIRE21Dataset',
    'NTIRE21ValDataset','NTIRE21v2Dataset',
    'NTIRE21v3Dataset','NTIRE21v3ValDataset',
    'ExtraDataset','ConcatDataset',
    'NTIRE21v3R3Dataset','NTIRE21v3R3ValDataset',
    'VideoTestHQR4NTIRE21Dataset',
    'MFQERGB','MFQERGBTestDataset',
    'MFQEv2HQDataset','VideoTestMFQEv2HQDataset',
    'MFQEv2BetaDataset','MFQEv2RTDataset',
    'VideoTestMFQEv2RTDataset'
    ]
