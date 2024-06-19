import pickle
import gzip

import numpy as np


"""
vrati MNIST dataset jako n-tice obsahujici trenovaci data, 
validacni data a testovaci data. Validacni data v me siti vubec nevyuzivam,
nicmene je to tak delane v oficialni dokumentaci MNIST pro vyuziti na zefektivneni site.
Tomuto zvysovani efektivnosti jsem se sice nevenoval, ale treba az pochopim matiku
za tim procesem, tak se na to vrhnu a je dobre pro to mit vse pripravene. Dataset je 
tedy rozdeleny na 50,000/10,000/10,000

Vse je vraceno jako n-tice o dvou hodnotach. Prvni obsahuje 
784-rozmernou ndarray odpovidajici pixelum nasich obrazku. 
Druha hodnota v n-tici je desetirozmenrna ndarray v niz je devet hodnot nula 
a jedna hodnota 1, reprezentujici spravnou odpoved 
"""

def nacti_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    trenovaci_data, validacni_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (trenovaci_data, validacni_data, test_data)

def nacti_data_format():

    """
    funkce pro lepsi formatovani datasetu... vse je zas delano podle oficialni
    dokumentace MNISTu a tak se tam znovu objevuje validacni set, kterz ale zustava
    nevyuzity.
    """

    tr_d, va_d, te_d = nacti_data()
    trenovaci_vstupy = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    trenovaci_vystupy = [odpoved_na_vektor(y) for y in tr_d[1]]
    trenovaci_data = zip(trenovaci_vstupy, trenovaci_vystupy)
    validacni_vstupy = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validacni_data = zip(validacni_vstupy, va_d[1])
    test_vstupy = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_vstupy, te_d[1])
    return (trenovaci_data, validacni_data, test_data)

def odpoved_na_vektor(j):
    #prevede cislo od 0 do 9 na jiz zmineny desetirozmerny vektor (ndarray)
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e