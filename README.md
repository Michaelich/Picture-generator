# Picture-generator

### Zagadnienie
MonaLisa - rekonstrukcja zadanego obrazu przy użyciu kół

### Przestrzeń poszukiwań

Osobnik reprezentowany jest przez macierz, w której każdy wiersz reprezentuje pojedyncze koło, a kolumny to parametry potrzebne do rysowania koła. 
Do natysowania koła używamy:
- współrzędna x na wynikowym obrazie, po skalowaniu wartości [0, szerokość obrazka]
- współrzędna y na wynikowym obrazie [0, wysokość obrazka] 
- Promień okręgu [10, 75]
- Wartości odpowiedzialne za kolor koła (RGB) [0, 255]
- Współczynnik przeźroczystości (alpha) [0,1]

Wszystkie parametry przyjmują w trakcie wykonywania obliczeń wartości (0,1], a następnie przy generowaniu obrazu są przeskalowane na odpowiednie wartości z podanych wyżej granic



### Funkcja celu
Funkcja celu generuje obraz zakodowany w osobniku, poprzez rysowanie kolejnych kół. Następnie wynikowy obraz porównywany jest z obrazem wejściowym przy pomocy ssim - structural similarity index measure, na każdym kolorze. Wynik jest uśredniany i sprowadzony do wartości procentowej.


### Użyte algorytmy
Algorytm który używamy to Evolution Strategy (ES), w którym odpowiednio dostosowaliśmy mutację oraz edycję sigm specjalnie pod problem. Tak jak zwykły ES nasz nie używa krzyżowania i polega jedynie na mutacjach

### Szczegółowy opis użytych metod
**Multiprocessing** - używamy obliczeń na wielu procesach żeby skrócić czas pracy algorytmu poprzez obliczanie powtarzalnych rzeczy na różnych procesach
**Dodawanie kółek** - mechanika dodawania kółek sprawia że pierwsze kółka na ogół przyjmują większe rozmiary i odpowiadają za tło natomiast nowo dodawane kółka mogą być użyte jako detale co przyspiesza znajdowanie coraz lepszych obrazków


### Szczegółowy opis uzyskanych wyników

Załączone końcowe wyniki przedstawiają najlepszego osobnika zapisanego co 2500 iteracji. Algorytm po 395000 iteracji przybliżał obraz w 58.21% (licząc według wspomnianej wyżej funkcji celu). Algorytm jednak nie utykał i do końca wprowadzał (co prawda drobne) poprawki. Brak fizycznego czasu nie pozwolił jednak na dłuższe przetestowanie algorytmu

### Wnioski końcowe, podsumowanie, perspektywy rozwoju.

Algorytm zauważalnie uczy się dostosowywać parametry kół, tak by dopasować się do wzorcowego obrazu. Wyniki uzyskane po około 400 000 iteracji osiągały rezultat strukturalnie przypominający Mona Lise. Algorytm miał jednak problem by dopasować dokładnie kolor tła. Dodatkowo opisany wynik otrzymany został po 1.5 dniach obliczeń.

Algorytm można by rozwinąć poprzez dodanie mutacji zmieniającej kolejność rysowania kół. Innym pomysłem jest zmniejszanie prawdopodobieństwa wyboru do mutacji kół znajdujących się na spodzie obrazu, bądź usuwanie ich by móc dodać nowe. Można także na sztywno zapisać wygenerowany z kół obraz i na nim rysować kolejne koła
