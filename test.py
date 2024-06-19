import mnist_formatovac

trenovaci_data, validacni_data, test_data = mnist_formatovac.nacti_data_format()
trenovaci_data = list(trenovaci_data)


import neuronova_sit

sit = neuronova_sit.Neuronova_sit([784, 30, 10])
sit.gradientni_sestup(trenovaci_data, 30, 10, 3.0, test_data=test_data)
#30 epoch, 10 velikost mini-batch, 3.0 velikost kroku gradientniho sestupu