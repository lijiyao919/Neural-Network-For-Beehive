from image_loader import *
import network

training_bee_data, training_no_bee_data, test_bee_data, test_no_bee_data, training_data, test_data = load_data_wrapper()


#print training_data
#print test_data
net = network.Network([1024, 30, 1])
net.SGD(training_data, 10, 10, 0.001, training_bee_data=training_bee_data,
                                     training_no_bee_data=training_no_bee_data,
                                     test_bee_data=test_bee_data,
                                     test_no_bee_data=test_no_bee_data,
                                     test_data=test_data)
net.save("network")