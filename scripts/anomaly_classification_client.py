#!/usr/bin/env python
import rospy
from birl_online_classification.srv import (
    BirlOnlineClassification,
    BirlOnlineClassificationRequest,
    BirlOnlineClassificationResponse,
)
import numpy as np

if __name__ == '__main__':
    rospy.init_node("anomaly_classification_client")

    matrix = np.matrix([[1,2,3], [3,4,5]])
    row_size = matrix.shape[0]
    col_size = matrix.shape[1]

    req = BirlOnlineClassificationRequest()
    req.timeseries_matrix.data = matrix.flatten().tolist()[0]
    req.timeseries_matrix.layout.dim = [row_size, col_size]
    resp = hmm_state_switch_proxy(req)

    rospy.loginfo(resp)
