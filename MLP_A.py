
import hickle as hkl
import numpy as np
import nnet as net
#import nnet_jit4 as net
import matplotlib.pyplot as plt 
from sklearn.model_selection import StratifiedKFold
        
class mlp_a_3w:
    def __init__(self, x, y_t, K1, K2, K3, lr, err_goal, \
                 disp_freq, ksi_inc, ksi_dec, er, max_epoch):
        self.x = x  # Przypisanie danych wejściowych do zmiennej instancyjnej
        self.L = self.x.shape[1]  # Liczba cech (kolumn) w danych wejściowych
        self.y_t = y_t  # Przypisanie docelowych etykiet do zmiennej instancyjnej
        self.K1 = K1  # Liczba neuronów w pierwszej warstwie ukrytej
        self.K2 = K2  # Liczba neuronów w drugiej warstwie ukrytej
        self.K3 = K3  # Liczba neuronów w warstwie wyjściowej
        self.lr = lr  # Współczynnik uczenia
        self.err_goal = err_goal  # Docelowy błąd SSE do osiągnięcia podczas treningu
        self.disp_freq = disp_freq  # Częstotliwość wyświetlania statusu podczas treningu
        self.ksi_inc = ksi_inc  # Współczynnik zwiększania współczynnika uczenia
        self.ksi_dec = ksi_dec  # Współczynnik zmniejszania współczynnika uczenia
        self.er = er  # Współczynnik do porównywania błędu SSE z poprzednią epoką
        self.max_epoch = max_epoch  # Maksymalna liczba epok treningu

        self.SSE_vec = []  # Lista do przechowywania wartości SSE w każdej epoce
        self.PK_vec = []  # Lista do przechowywania wartości dokładności w każdej epoce

        self.w1, self.b1 = net.nwtan(self.K1, self.L)  # Inicjalizacja wag i biasów dla pierwszej warstwy ukrytej
        self.w2, self.b2 = net.nwtan(self.K2, self.K1)  # Inicjalizacja wag i biasów dla drugiej warstwy ukrytej
        self.w3, self.b3 = net.rands(self.K3, self.K2)  # Inicjalizacja wag i biasów dla warstwy wyjściowej

        self.SSE = 0  # Inicjalizacja sumy kwadratów błędów na 0
        self.lr_vec = list()  # Lista do przechowywania wartości współczynnika uczenia w każdej epoce

    def predict(self, x):  # Definicja metody predict, która przyjmuje dane wejściowe x
        n = np.dot(self.w1, x)  # Oblicza sumę ważoną dla pierwszej warstwy ukrytej (mnożenie wag przez dane wejściowe)
        self.y1 = net.tansig(n, self.b1 * np.ones(
            n.shape))  # Stosuje funkcję aktywacji tansig (tangens hiperboliczny) z dodanymi biasami dla pierwszej warstwy
        n = np.dot(self.w2,
                   self.y1)  # Oblicza sumę ważoną dla drugiej warstwy ukrytej (mnożenie wag przez wyjścia z pierwszej warstwy)
        self.y2 = net.tansig(n, self.b2 * np.ones(
            n.shape))  # Stosuje funkcję aktywacji tansig z dodanymi biasami dla drugiej warstwy
        n = np.dot(self.w3,
                   self.y2)  # Oblicza sumę ważoną dla warstwy wyjściowej (mnożenie wag przez wyjścia z drugiej warstwy)
        self.y3 = net.purelin(n, self.b3 * np.ones(
            n.shape))  # Stosuje funkcję aktywacji purelin (funkcja liniowa) z dodanymi biasami dla warstwy wyjściowej
        return self.y3  # Zwraca wyjścia warstwy wyjściowej jako wynik prognozy sieci neuronowej

    def train(self, x_train,
              y_train):  # Metoda treningowa, która aktualizuje wagi sieci neuronowej na podstawie danych treningowych
        for epoch in range(1, self.max_epoch + 1):  # Pętla po epokach treningowych
            self.y3 = self.predict(x_train)  # Generowanie prognoz dla danych treningowych przy użyciu metody predict
            self.e = y_train - self.y3  # Obliczenie błędu prognozowania dla danych treningowych

            self.SSE_t_1 = self.SSE  # Zapisanie wartości błędu sumy kwadratów z poprzedniej epoki
            self.SSE = net.sumsqr(self.e)  # Obliczenie błędu sumy kwadratów dla aktualnej epoki
            self.PK = (1 - sum((abs(self.e) >= 0.5).astype(int)[0]) / self.e.shape[
                1]) * 100  # Obliczenie dokładności prognozowania w procentach
            self.PK_vec.append(self.PK)  # Dodanie dokładności prognozowania do listy

            if self.SSE < self.err_goal or self.PK == 100:  # Sprawdzenie warunków zakończenia treningu (osiągnięcie celu błędu lub 100% dokładności)
                break  # Zakończenie treningu

            if np.isnan(self.SSE):  # Sprawdzenie, czy błąd sumy kwadratów jest wartością NaN
                break  # Zakończenie treningu

            else:  # Aktualizacja współczynnika uczenia na podstawie poprzednich błędów
                if self.SSE > self.er * self.SSE_t_1:  # Jeśli błąd zwiększył się w stosunku do poprzedniej epoki
                    self.lr *= self.ksi_dec  # Zmniejszenie współczynnika uczenia
                elif self.SSE < self.SSE_t_1:  # Jeśli błąd zmniejszył się w stosunku do poprzedniej epoki
                    self.lr *= self.ksi_inc  # Zwiększenie współczynnika uczenia
            self.lr_vec.append(self.lr)  # Dodanie aktualnego współczynnika uczenia do listy

            # Obliczenie delty dla każdej warstwy na podstawie błędu
            self.d3 = net.deltalin(self.y3, self.e)  # Delta dla warstwy wyjściowej
            self.d2 = net.deltatan(self.y2, self.d3, self.w3)  # Delta dla drugiej warstwy ukrytej
            self.d1 = net.deltatan(self.y1, self.d2, self.w2)  # Delta dla pierwszej warstwy ukrytej

            # Obliczenie zmian wag i biasów dla każdej warstwy na podstawie delty i współczynnika uczenia
            self.dw1, self.db1 = net.learnbp(self.x.T, self.d1,
                                             self.lr)  # Zmiany wag i biasów dla pierwszej warstwy ukrytej
            self.dw2, self.db2 = net.learnbp(self.y1, self.d2,
                                             self.lr)  # Zmiany wag i biasów dla drugiej warstwy ukrytej
            self.dw3, self.db3 = net.learnbp(self.y2, self.d3, self.lr)  # Zmiany wag i biasów dla warstwy wyjściowej

            # Aktualizacja wag i biasów dla każdej warstwy
            self.w1 += self.dw1  # Aktualizacja wag dla pierwszej warstwy ukrytej
            self.b1 += self.db1  # Aktualizacja biasów dla pierwszej warstwy ukrytej
            self.w2 += self.dw2  # Aktualizacja wag dla drugiej warstwy ukrytej
            self.b2 += self.db2  # Aktualizacja biasów dla drugiej warstwy ukrytej
            self.w3 += self.dw3  # Aktualizacja wag dla warstwy wyjściowej
            self.b3 += self.db3  # Aktualizacja biasów dla warstwy wyjściowej

            self.SSE_vec.append(self.SSE)  # Dodanie błędu sumy kwadratów do listy


x, y_t, x_norm, x_n_s, y_t_s = hkl.load(
    'hepatitis.hkl')  # Wczytanie danych z pliku 'hepatitis.hkl' przy użyciu biblioteki hickle

# Ustawienie parametrów dla treningu sieci neuronowej
max_epoch = 2000  # Maksymalna liczba epok treningu
err_goal = 0.25  # Docelowy błąd SSE do osiągnięcia podczas treningu
disp_freq = 10  # Częstotliwość wyświetlania statusu podczas treningu
lr = 1e-4  # Początkowy współczynnik uczenia
ksi_inc = 1.05  # Współczynnik zwiększania współczynnika uczenia
ksi_dec = 0.7  # Współczynnik zmniejszania współczynnika uczenia
er = 1.04  # Współczynnik do porównywania błędu SSE z poprzednią epoką
K1 = 7  # Liczba neuronów w pierwszej warstwie ukrytej
K2 = 3  # Liczba neuronów w drugiej warstwie ukrytej

# Przygotowanie danych
data = x_n_s.T  # Transpozycja znormalizowanych danych wejściowych
target = y_t_s  # Docelowe etykiety

# Ustalenie liczby neuronów w warstwie wyjściowej
K3 = target.shape[0] if len(
    target.shape) > 1 else 1  # Jeśli target jest macierzą, ustaw K3 jako liczbę wierszy, w przeciwnym razie ustaw 1

# Ustawienia dla walidacji krzyżowej
CVN = 10  # Liczba podziałów (folds) dla walidacji krzyżowej
skfold = StratifiedKFold(n_splits=CVN)  # Inicjalizacja stratified k-fold cross-validatora
PK_vec = np.zeros(CVN)  # Inicjalizacja wektora do przechowywania dokładności dla każdego podziału

# Pętla dla każdej iteracji walidacji krzyżowej
for i, (train, test) in enumerate(skfold.split(data, np.squeeze(target)), start=0):
    x_train, x_test = data[train], data[test]  # Podział danych na zbiór treningowy i testowy
    y_train, y_test = np.squeeze(target)[train], np.squeeze(target)[
        test]  # Podział etykiet na zbiór treningowy i testowy

    # Inicjalizacja sieci neuronowej z aktualnym podziałem danych
    mlpnet = mlp_a_3w(x_train, y_train, K1, K2, K3, lr, err_goal, disp_freq, ksi_inc, ksi_dec, er, max_epoch)

    # Trening sieci neuronowej
    mlpnet.train(x_train.T, y_train.T)

    # Generowanie prognoz dla zbioru testowego
    result = mlpnet.predict(x_test.T)

    # Obliczenie liczby próbek w zbiorze testowym
    n_test_samples = test.size

    # Obliczenie dokładności klasyfikacji dla aktualnego podziału
    PK_vec[i] = sum((abs(result - y_test) < 0.5).astype(int)[0]) / n_test_samples * 100

    # Wyświetlenie wyników dla aktualnego podziału
    print("Test #{:<2}: PK_vec {} test_size {}".format(i, PK_vec[i], n_test_samples))

# Obliczenie średniej dokładności dla wszystkich podziałów
PK = np.mean(PK_vec)
print("PK {}".format(PK))

# Wykres współczynnika uczenia w funkcji epok treningowych
plt.figure()
plt.plot(mlpnet.lr_vec)
plt.title("lr=f(epoch)")

# Wykres sumy kwadratów błędów w funkcji epok treningowych
plt.figure()
plt.plot(mlpnet.SSE_vec)
plt.title("SSE=f(epoch)")
