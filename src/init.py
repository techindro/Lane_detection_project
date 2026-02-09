"""
Lane Detection System Package
"""

__version__ = "1.0.0"
__author__ = "Shubham Patel"
__email__ = "shubhamkumarpatel45@gmail.com"

from .config import config
from .pipeline import LaneDetectionPipeline
from .main import LaneDetectionSystem

__all__ = ['config', 'LaneDetectionPipeline', 'LaneDetectionSystem']
