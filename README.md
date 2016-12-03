# rn-project
Neural Networks Project


## Description

Source: [profs.info.uaic.ro/~rbenchea/proiecte_rn](http://profs.info.uaic.ro/~rbenchea/proiecte_rn.html)


>Breakout: 50p
Realizati un agent care trebuie sa invete sa joace Breakout. Jocul Breakout, https://gym.openai.com/envs/Breakout-v0, presupune aruncarea unei mingi spre un perete unde sunt mai multe caramizi. Orice caramida este sparta atunci cand este atinsa de minge. Mingea este directionata printro platforma aflata la baza, iar directia este data de unghiul pe care-l face cu platforma. Pot fi folositi si peretii laterali sau cel superior. Jocul se termina atunci cand sunt sparte toate caramizile sau cand mingea cade sub platforma. Mediul de lucru trebuie sa vi-l creati voi. Scorurile (pozitiv sau negativ) precum si informatiile primite de la mediu, vi le setati voi, in functie de cum e mai convenabil.

>Variante: 
+ mediu descarcat de pe internet, dar antrenament bazat pe pixeli: 45
+ mediu descarcat de pe internet, dar antrenament bazat pe informatii despre mediu extrase de voi: 35p
+ mediu descarcat de pe internet, dar antrenament bazat pe informatii despre mediu incluse in proiectul gasit pe internet: 30p


## Installation

Dependencies:
+ `pip install keras` / `pip install tensorflow`
  + prereq: `sudo apt-get install python-numpy`
+ `pip install gym`
+ `pip install gym[atari]`
  + prereq: `sudo apt-get install python-dev zlib1g-dev`
+ `??` - possibly needed dependencies: `sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose`
