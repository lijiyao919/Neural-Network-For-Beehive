import cv2
import network
import os
import numpy as np

dirpath = "./testNet"
test = []

#read test photo in testNet file
for root, dirs, files in os.walk(dirpath):
    for item in files:
        if item.endswith('.png'):
            ip = os.path.join(root, item)
            img = cv2.imread(ip)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = (img.flatten() / float(255))
            test.append(list(img))

test = np.array(test)
test_inputs = [np.reshape(x, (1024, 1)) for x in test]

#file network is to persisit weights and biases
net=network.restore("network")

print "The result:"
for test in test_inputs:
    a=net.feedforward(test)
    if abs(a-0) < 0.5:
        print "No Bee"
    else:
        print "Has Bee"
