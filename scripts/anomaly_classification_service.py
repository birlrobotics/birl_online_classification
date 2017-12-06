#!/usr/bin/env python
import rospy
from birl_online_classification.srv import (
    BirlOnlineClassification,
    BirlOnlineClassificationResponse,
)
from birl_online_classification import anomaly_classifier
import numpy as np

def service_cb(req):
    flatten_matrix = req.timeseries_matrix.data
    row_size = req.timeseries_matrix.layout.dim[0] 
    col_size = req.timeseries_matrix.layout.dim[1] 

    matrix = np.matrix(flatten_matrix).reshape((row_size, col_size))
 
    label, confidence = anomaly_classifier.run(matrix)

    resp = BirlOnlineClassificationResponse()
    resp.label = label
    resp.confidence = confidence

    rospy.loginfo(matrix)
    rospy.loginfo(label, confidence)
    
    return resp

if __name__ == '__main__':
    rospy.init_node("anomaly_classification_service")
    oc = rospy.Service('hmm_state_switch', State_Switch, state_switch_handle)
    rospy.spin()
