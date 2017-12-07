#!/usr/bin/env python
import rospy
from birl_online_classification.srv import (
    BirlOnlineClassification,
    BirlOnlineClassificationRequest,
    BirlOnlineClassificationResponse,
)
from std_msgs.msg import MultiArrayDimension
import numpy as np
import pandas as pd
import os

if __name__ == '__main__':
    rospy.init_node("anomaly_classification_client")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(dir_path, "test_label0.csv"), sep=',')
    matrix = df.values[:, 1:]
    row_size = matrix.shape[0]
    col_size = matrix.shape[1]

    req = BirlOnlineClassificationRequest()
    req.timeseries_matrix.data = matrix.flatten().tolist()

    mad = MultiArrayDimension()
    mad.size = row_size
    mad.stride = col_size
    req.timeseries_matrix.layout.dim = [mad]

    ocp = rospy.ServiceProxy('BirlOnlineClassification', BirlOnlineClassification)

    resp = ocp(req)

    print resp
