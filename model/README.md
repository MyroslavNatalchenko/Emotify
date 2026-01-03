# Ewolucja Modelu Multimodalnego (MERT + CNN)

Poniżej znajduje się krótki opis 5 etapów ulepszania modelu do klasyfikacji nastroju i gatunków muzycznych.

### Etap 1: Fuzja multimodalna (Punkt wyjścia)

Na początku połączyliśmy dwie niezależne gałęzie:

1. **CNN** trenowaną od zera na Mel-spektrogramach.
2. **MLP** przyjmującą gotowe wektory z modelu MERT.
Celem było stworzenie modelu, który widzi zarówno obraz dźwięku, jak i jego semantykę.

---

### Etap 2: Walka z nierównowagą klas

Model miał niską pewność siebie z powodu nierównych danych (dużo "pop", mało "space").

* **Zmiany:** Dodaliśmy agresywne wagi klas (`pos_weight = neg/pos`), stratyfikację danych (`iterative-stratification`) oraz dobieranie progów (`threshold tuning`).
* **Wynik:** Model zaczął "krzyczeć" rzadkie tagi. Wystąpiło ogromne **przeuczenie (overfitting)** – `Val Loss` poszybował w górę, a model dawał zbyt dużo fałszywych pozytywów (wysokie ROC-AUC, niskie PR-AUC).

---

### Etap 3: Korekta przeuczenia (Soft Weights & Dropout)

Musieliśmy "uspokoić" model, który zbyt agresywnie reagował na kary za błędy.

* **Zmiany:** Złagodziliśmy wagi (użycie pierwiastka `sqrt`), zwiększyliśmy **Dropout** (z 0.3 na 0.5), aby utrudnić zapamiętywanie danych.
* **Wynik:** Przeuczenie zniknęło (krzywe Loss idą w dół), ale model stał się **zbyt ostrożny (under-confident)**. Pewność predykcji spadła drastycznie, rzadkie gatunki przestały być wykrywane.

---

### Etap 4: Wdrożenie Focal Loss

Zamiast ręcznie manipulować wagami, zastosowaliśmy mądrzejszą funkcję straty.

* **Zmiany:** Zastąpienie `BCEWithLogitsLoss` przez **Focal Loss** (`gamma=2`). Funkcja ta automatycznie skupia się na trudnych przykładach, ignorując te łatwe.
* **Wynik:** Stabilniejszy trening. Model przestał sztucznie zawyżać pewność, ale wymagał użycia indywidualnych progów (thresholds) dla każdego gatunku przy testowaniu.

---

### Etap 5: Architektura "Zaufaj MERT" (Obecny)

Zauważyliśmy, że trenowana od zera sieć CNN wprowadzała szum, który zagłuszał pre-trenowaną wiedzę modelu MERT.

* **Zmiany:**
* **MERT:** Zwiększenie liczby cech (256/512 neuronów) i niski Dropout (0.2).
* **CNN:** Drastyczne zmniejszenie "władzy" tej gałęzi (redukcja do 32 cech).


* **Cel:** Wymuszenie na modelu, by opierał decyzje głównie na "doświadczonym" modelu MERT, traktując spektrogramy tylko jako wsparcie.