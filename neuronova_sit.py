import random
import numpy as np

class Neuronova_sit(object):

    def __init__(self, struktura):
        self.pocet_vrstev = len(struktura)
        self.struktura = struktura #muze vypadat napr. jako [5,2,1], coz by vytvorilo sit s peti neurony v prvni vrsve, dvemi v druhe a jednim v posledni 
        self.prahy = [np.random.randn(y, 1) for y in struktura[1:]]#nahodne vytvori hodnoty prahu (1 kvuli tomu, aby nebyly vytvoreny v prvni vrstve)
        self.vahy = [np.random.randn(y, x)
                        for x, y in zip(struktura[:-1], struktura[1:])]

    def neuron_akt_hod(self, a): #rovnice podle ktere neurony pocitaji aktivacni hodnotu a
        for p, v in zip(self.prahy, self.vahy):
            a = sigmoid(np.dot(v, a)+p)
        return a

    def gradientni_sestup(self, trenovaci_data, epochy, mini_batch_velikost, rychlost_uceni,
            test_data=None):

        trenovaci_data = list(trenovaci_data) #list n-tic (x,y) reprezentujici vstupy a ocekavane vystupy
        n = len(trenovaci_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochy): #nahodne promicha testovaci data, rozdeli je na mini-batche 
            random.shuffle(trenovaci_data)
            mini_batches = [
                trenovaci_data[k:k+mini_batch_velikost]
                for k in range(0, n, mini_batch_velikost)]
            for mini_batch in mini_batches:#aplikuje jeden krok gradientniho sestupu podle funkce update_mini_batch
                self.update_mini_batch(mini_batch, rychlost_uceni)
            
            if test_data: #prubezne vypise jakou ma sit uspesnost
                print (f"Epocha {j}: {self.vyhodnot(test_data)}/{n_test}")
            else:
                print(f"Epocha {j} - konec")

    def update_mini_batch(self, mini_batch, rychlost_uceni): #Zmeni hodnoty vah a prahu aplikaci gradientniho sestupu vypocitaneho ve funkci alg_zpetneho_sireni

        nabla_p = [np.zeros(p.shape) for p in self.prahy] #nazvy nabla a delta vychazi ze standartniho znaceni 
        nabla_v = [np.zeros(v.shape) for v in self.vahy]
        for x, y in mini_batch:
            delta_nabla_p, delta_nabla_v = self.alg_zpetneho_sireni(x, y) #(nabla_v, nabla_p) je gradient chybove funkce
            nabla_p = [np+dnp for np, dnp in zip(nabla_p, delta_nabla_p)] #np = nabla prahu, dnp = delta nabla prahu (obdobne s vahami)
            nabla_v = [nv+dnv for nv, dnv in zip(nabla_v, delta_nabla_v)]
        self.vahy = [v-(rychlost_uceni/len(mini_batch))*nv
                        for v, nv in zip(self.vahy, nabla_v)]
        self.prahy = [p-(rychlost_uceni/len(mini_batch))*np
                       for p, np in zip(self.prahy, nabla_p)]

    def alg_zpetneho_sireni(self, x, y): #Vrati n-tici (nabla_v, nabla_p) reprezentujici gradient chybove funkce
        nabla_p = [np.zeros(p.shape) for p in self.prahy]
        nabla_v = [np.zeros(v.shape) for v in self.vahy]

        aktivace = x
        aktivace_list = [x] # list obsahujici vsechny aktivace, hezky vrstvu po vrstve
        zs = [] #stejne jako v teoreticke casti Z odpovida aktivacni hodnote neuronu
        for p, v in zip(self.prahy, self.vahy):
            z = np.dot(v, aktivace)+p
            zs.append(z)
            aktivace = sigmoid(z)
            aktivace_list.append(aktivace)
        # zpetne sireni
        delta = self.derivace_chyby(aktivace_list[-1], y) * \
            sigmoid_derivace(zs[-1])
        nabla_p[-1] = delta
        nabla_v[-1] = np.dot(delta, aktivace_list[-2].transpose())

        for l in range(2, self.pocet_vrstev): #l znaci vrstvu napr. l=2 znaci predposledni vrstvu atd.
            z = zs[-l]
            sd = sigmoid_derivace(z)
            delta = np.dot(self.vahy[-l+1].transpose(), delta) * sd
            nabla_p[-l] = delta
            nabla_v[-l] = np.dot(delta, aktivace_list[-l-1].transpose())
        return (nabla_p, nabla_v)



    def vyhodnot(self, test_data): #vrati pocet uspesnych rozpoznani
        test_vysledky = [(np.argmax(self.neuron_akt_hod(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_vysledky)

    def derivace_chyby(self, output_aktivace_list, y): #vrati vektor parcialnich derivaci chybove funkce a a
        return (output_aktivace_list-y)

# sigmoidni funkce a jeji derivace
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_derivace(z):
    return sigmoid(z)*(1-sigmoid(z))