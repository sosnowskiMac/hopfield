# Pamięć skojarzeniowa – sieć Hopfielda (reguła Hebba i reguła Oji)

## Struktura

```
hopfield/network.py      sieć: trening (Hebb, Oja), dynamika sync/async, energia
hopfield/data.py         wczytywanie CSV, uszkadzanie wejść, statystyki zbiorów
hopfield/viz.py          bitmapy, kolejne kroki zbiegania, animacje GIF, energia
hopfield/experiments.py  eksperymenty do raportu
run.py                   CLI
prepare_images.py        zdjęcia -> CSV z wartościami 1/-1 (duże bitmapy)
data/                    tu wrzucić dostarczone zbiory *.csv
```

## Założenia o danych

* CSV: jeden wiersz = jeden wzorzec (spłaszczony wierszami), wartości `1` i `-1`.
* Kształt `wiersze x kolumny`: z nazwy pliku (`cokolwiek_25x50.csv` → 25 wierszy, 50 kolumn),
  w przeciwnym razie kwadrat. Jeśli plik ma odwrotną orientację, zmień nazwę pliku.

## Instalacja

```
pip install -r requirements.txt
```

## Konfiguracja sieci

```python
from hopfield.network import HopfieldNetwork
net = HopfieldNetwork((25, 25))                  # liczba neuronów = wiersze * kolumny
net.train(X, "hebb")                             # lub "oja" (solver="iterative"|"svd", eta, epochs, zero_diag)
r = net.recall(x0, mode="sync")                  # lub "async"
r.history, r.status, r.steps, r.period           # status: stable | cycle | max_steps
```

Wagi są trzymane w postaci niskiego rzędu `W = A·Bᵀ − diag(d)`, więc bitmapy 200×300 (n = 60000)
nie wymagają macierzy 60000×60000. Pamięć i czas kroku to O(n·p).

* **Hebb**: `W = (1/n) Σ ξξᵀ`, `w_ii = 0`.
* **Oja** (reguła podprzestrzeni): `Q ∈ R^{n×k}`, `y = Qᵀx`, `ΔQ = η (x − Q y) yᵀ`, `W = Q Qᵀ`.
  `k` = rząd macierzy wzorców. Domyślnie diagonala zachowana (`W` jest wtedy rzutnikiem);
  `--oja-zero-diag` ją zeruje.
* `--oja-solver svd` (domyślny w eksperymentach) liczy granicę reguły Oji bezpośrednio z SVD
  (ortonormalna baza `span(X)`). Jest szybki przy przeglądach po `p`.
  `--oja-solver iterative` uruchamia właściwą pętlę uczącą; `python run.py check-oja plik.csv`
  porównuje oba warianty.
* Aktualizacja: `x_i ← sign(Σ_j w_ij x_j)`, przy polu równym 0 neuron zachowuje stan.
  Sync: wszystkie neurony naraz, wykrywane są cykle. Async: losowa permutacja neuronów w każdym przebiegu,
  koniec po przebiegu bez zmian.

## Polecenia → zagadnienia z zadania

| Zagadnienie | Polecenie | Wyniki w `output/` |
|---|---|---|
| Skuteczność reguł, liczba stabilnych wzorców, które zbiory się uczą | `python run.py stability` | `stability/stability.csv`, `stable_vs_p.png`, `correlation_*.png`, `patterns_*.png` |
| Odzyskiwanie z uszkodzonego wejścia (szum, wycięty blok) vs liczność i rodzaj zbioru | `python run.py recovery` | `recovery/recovery_*.png/csv`, `recovery_vs_p.png` |
| 25×25 vs 25×50 (podobne wzory) | `python run.py compare data/A_25x25.csv data/B_25x50.csv` | `compare/` + statystyki (obciążenie `p/n`, korelacje, bias, rząd) |
| Zbiór z grupami podobnych vs zróżnicowany, podobna liczność | `python run.py compare A.csv B.csv --equalize` | `compare/` |
| Losowe wejście | `python run.py random` | `random_input/` (wzorzec / negatyw / stan pozorny / cykl) |
| Największy zbiór 5×5 ze wszystkimi wzorcami stabilnymi | `python run.py stable-set` | `stable_set/` |
| Oscylacja między dwoma stanami | `python run.py oscillation` | `oscillation/` |
| Duże bitmapy ≥ 200×300 | `python prepare_images.py koty/ --height 300 --width 200` potem `python run.py large data_large/cats_300x200.csv --modes sync` | `large/` |
| Kolejne kroki zbiegania (jeden przypadek) | `python run.py demo plik.csv --rule oja --mode async --kind block --level 0.3 [--show]` | `demo/` (PNG klatek, energia, GIF) |

Wspólne opcje: `--trials`, `--reps`, `--p-points`, `--noise`, `--modes sync async`, `--out`, `--seed`.
Dla bitmap 60000 neuronów tryb async jest wolny (pętla po neuronach), więc dla `large` zalecane jest
`--modes sync` albo mała liczba `--trials`.

## Dlaczego te reguły działają (materiał do raportu)

**Dynamika i energia.** Dla symetrycznej `W` i `w_ii = 0` funkcja `E(x) = −½ xᵀWx` nie rośnie przy
asynchronicznej aktualizacji (Hopfield 1982), a przy symetrycznej `W` z `w_ii ≥ 0` także. Stany
stabilne to lokalne minima `E`, więc pamięć skojarzeniowa to „zjeżdżanie” z uszkodzonego wejścia do
najbliższego minimum. Przy synchronicznej aktualizacji sieć o symetrycznych wagach ma cykle o okresie
co najwyżej 2 (Goles, Fogelman-Soulié, Olivos 1985). Dlatego możliwa jest oscylacja dwustanowa, a nie
dłuższa.

**Hebb.** `E(x) = −(1/2n) Σ_μ (ξ^μ·x)²`: każdy wzorzec wykopuje studnię, bo jego nakładanie `ξ^μ·x`
jest największe dla `x = ξ^μ`. Pole na neuronie dla wzorca ν:
`h_i = ξ_i^ν (1 − p/n) + (1/n) Σ_{μ≠ν} ξ_i^μ (ξ^μ·ξ^ν)`. Pierwszy składnik to sygnał, drugi to szum
przesłuchów (crosstalk). Dla losowych nieskorelowanych wzorców szum ma wariancję ≈ p/n, więc wzorce są
stabilne, dopóki `p ≲ 0.138 n` (Amit, Gutfreund, Sompolinsky 1985). Dla bezbłędnego odzyskania
wszystkich bitów granica jest niższa, ok. `n / (2 ln n)` (McEliece i in. 1987). Dla wzorców skorelowanych
(podobnych, z niezerowym biasem) przesłuchy rosną, ponieważ `ξ^μ·ξ^ν` jest duże, i wzorce przestają być
stabilne. Dla wzorców ortogonalnych przesłuchów nie ma. Dodatkowo powstają stany pozorne: negatywy
wzorców (`E(x) = E(−x)`), mieszaniny nieparzystej liczby wzorców i stany typu szkło spinowe.

**Oja.** Reguła Oji (1982, wersja podprzestrzenna 1989) dodaje do składnika hebbowskiego człon
normalizujący `−y yᵀ Q`. Q zbiega do ortonormalnej bazy podprzestrzeni rozpiętej przez wzorce
(główne wektory własne macierzy korelacji), więc `W = QQᵀ` to rzutnik ortogonalny `P` na `span{ξ^μ}`.
Wzorzec należący do podprzestrzeni spełnia `Pξ = ξ`, więc jest stały dla dowolnych skorelowanych
wzorców, o ile są liniowo niezależne. To analog reguły projekcyjnej / pseudoodwrotności
(Personnaz i in. 1985; Kanter i Sompolinsky 1987), która usuwa przesłuchy. Ograniczenia: pojemność to
rząd macierzy wzorców ≤ n. Przy zachowanej diagonali, a przy `p → n` `P → I` i każdy stan jest
„stabilny”, czyli obszary przyciągania znikają (stąd wariant `--oja-zero-diag`, por. Kanter–Sompolinsky).
Zbiór liniowo zależny daje rząd mniejszy niż `p`. Wzorzec poza podprzestrzenią (np. dopełnienie
wektorów niezależnych) może nie być stabilny po `sign`.

## Wskazówki do interpretacji wyników

* Stabilność wzorca ≠ dobre odzyskiwanie. Sprawdź jednocześnie `stable_frac` i `recovery_*`.
* 25×25 vs 25×50: porównaj `p/n` (n różni się dwukrotnie), `mean_abs_overlap`, `bias`, `rank` z `compare_stats_*.csv`.
* Oscylacja: wzorce i wejście zapisywane są do CSV w `output/oscillation/`. Gwarantowany przykład analityczny
  to 2 neurony, wzorzec `(1,−1)`, wejście `(1,1)` (sync: `(1,1) ↔ (−1,−1)`, async: zbiega).
* Zbiór 5×5 (`stable-set`): ortogonalnych wektorów ±1 długości 25 nie ma (iloczyn dwóch takich wektorów jest
  nieparzysty), więc dla Hebba szukany jest zbiór o możliwie małych nakładaniach. Dla Oji
  z diagonalą wynik jest ograniczony tylko `--max-size` (to przypadek `P = I`, patrz wyżej).
  Wariant bez diagonali pokazuje nietrywialną granicę.

## Literatura

Hopfield 1982 (PNAS); Amit, Gutfreund, Sompolinsky 1985 (PRL); McEliece, Posner, Rodemich, Venkatesh 1987;
Oja 1982 (J. Math. Biol.), Oja 1989; Personnaz, Guyon, Dreyfus 1985; Kanter, Sompolinsky 1987;
Goles, Fogelman-Soulié, Olivos 1985; Hertz, Krogh, Palmer, *Introduction to the Theory of Neural Computation*.
