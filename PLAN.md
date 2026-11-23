# Plan implementacji (dla LLM / agenta)

Cel: projekt pamięci skojarzeniowej Hopfielda (Hebb, Oja; sync/async; wizualizacja bitmap; eksperymenty do raportu).
Zasady: prosty kod, minimum komentarzy, Python 3 + numpy/pandas/matplotlib/pillow. Dane: CSV z `1` i `-1`,
jeden wiersz = jeden wzorzec, kształt z nazwy pliku `HxW` albo kwadrat.

## Kolejność prac (każdy etap kończy się spełnieniem kryterium)

1. **`hopfield/network.py`** – klasa `HopfieldNetwork(shape, seed)`.
   * `train(X, rule, zero_diag, **kw)`; wagi w postaci `A @ Bt.T − diag(d)` (niski rząd, n do 60000).
   * Hebb: `A = Xᵀ`, `Bt = Xᵀ/n`, diagonala zerowana. Oja: iteracyjna reguła podprzestrzeni `ΔQ = η(x − Qy)yᵀ`,
     opcjonalnie `solver="svd"`; `A = Bt = Q`, diagonala domyślnie zachowana.
   * `recall(x0, mode, max_steps, record_every)` → `RecallResult(history, status, steps, period)`.
     Sync: wykrywanie cykli (`stable`/`cycle`/`max_steps`). Async: losowa permutacja, przyrostowa aktualizacja `h = x@Bt`.
   * `stable_mask`, `energy`, `field`, `update`, `set_state`.
   * Kryterium: pole 0 → neuron zachowuje stan; `weights()` dla małych n zgadza się z Hebbem `XᵀX/n − diag`.
2. **`hopfield/data.py`** – `load_patterns`, `parse_shape`, `corrupt` (`noise`, `block`), `subset`, `pattern_stats`.
3. **`hopfield/viz.py`** – `draw`, `show_grid`, `plot_history`, `plot_energy`, `animate` (GIF lub okno).
   * Kryterium: kolejne kroki zbiegania widoczne zarówno dla sync (klatka = krok), jak i async (klatka = co `record_every` aktualizacji).
4. **`hopfield/experiments.py`** – po jednej funkcji na zagadnienie (patrz tabela w README), wyniki do `output/<nazwa>/` jako PNG + CSV.
5. **`run.py`** – CLI z podpoleceniami; **`prepare_images.py`** – zdjęcia → CSV (skala szarości, przycięcie, próg = mediana).
6. **Uruchomienie eksperymentów (po stronie użytkownika)** na dostarczonych zbiorach + `check-oja`.
7. **Raport** (poza kodem) wg szkieletu poniżej.

## Zagadnienia → funkcje → co raportować

| Zagadnienie | Funkcja | Metryki / wykresy |
|---|---|---|
| Skuteczność reguł, liczba stabilnych | `exp_stability` | `stable_frac` vs `p`, linia `0.138 n`, macierze korelacji |
| Które zbiory się uczą, czemu nie | `exp_stability` | `mean_abs_overlap`, `bias`, `rank` vs `p`; odwołanie do crosstalk / liniowej niezależności |
| 25×25 vs 25×50 | `exp_compare` | `p/n`, korelacje, bias, recovery |
| Odzyskiwanie vs liczność/rodzaj | `exp_recovery` | `success`, `overlap` vs poziom szumu/bloku; `recovery_vs_p` |
| Podobne grupy vs zróżnicowane | `exp_compare --equalize` | jak wyżej przy równym `p` |
| Losowe wejście | `exp_random_input` | udział: wzorzec, negatyw, stan pozorny, cykl; przykładowe stany końcowe |
| Zbiór 5×5 wszystkie stabilne | `exp_stable_set` | rozmiar zbioru dla Hebb / Oja / Oja bez diagonali; obrazek zbioru |
| Oscylacja | `exp_oscillation` | przykład 2 neurony + znaleziony zbiór 5×5 i wejście, ścieżka sync vs async |
| Duże bitmapy | `exp_large` | stabilność i recovery dla `p ∈ {2,5,10,20}`, przykładowe klatki zbiegania |

## Szkielet raportu

1. Model: dynamika, energia, sync vs async (cykle ≤ 2).
2. Reguły uczenia: wzory i **dlaczego działają** (Hebb: studnie energii + crosstalk, pojemność ≈ 0.138n; Oja: zbieżność do rzutnika
   na `span(X)`, stabilność wzorców liniowo niezależnych, związek z regułą projekcyjną). Źródła w README.
3. Metodyka eksperymentów (liczba prób, uszkodzenia, kryterium sukcesu: dokładne odzyskanie oryginału).
4. Wyniki wg tabeli powyżej, każdy z interpretacją odniesioną do literatury.
5. Dyskusja ograniczeń (stany pozorne, negatywy, diagonala w Oja, koszt async dla dużych n).

## Uwagi dla agenta wykonującego

* Nie modyfikuj interfejsu `HopfieldNetwork` bez aktualizacji `experiments.py` (korzysta z `A`, `Bt`, `d`).
* Nie uruchamiaj `iterative` Oji w pętlach po wielu `p` bez potrzeby (wolne); domyślnie `svd` + jednorazowy `check-oja`.
* Gdy nazwy plików nie zawierają `HxW` a `n` nie jest kwadratem, `parse_shape` zgłosi błąd – dopisz wymiary do nazwy pliku.
