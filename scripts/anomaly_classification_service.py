#!/usr/bin/env python
import rospy
from birl_online_classification.srv import (
    BirlOnlineClassification,
    BirlOnlineClassificationResponse,
)
from birl_online_classification import anomaly_classifier
import numpy as np
import ipdb

def service_cb(req):
    flatten_matrix = req.timeseries_matrix.data
    row_size = req.timeseries_matrix.layout.dim[0].size
    col_size = req.timeseries_matrix.layout.dim[0].stride 
    matrix = np.matrix(flatten_matrix).reshape((row_size, col_size))

    print matrix
 
    label, confidence = anomaly_classifier.run(matrix)

    resp = BirlOnlineClassificationResponse()
    resp.label.data = label
    resp.confidence.data = confidence
    
    return resp

if __name__ == '__main__':
    rospy.init_node("anomaly_classification_service")
    oc = rospy.Service('BirlOnlineClassification', BirlOnlineClassification, service_cb)
    rospy.spin()
