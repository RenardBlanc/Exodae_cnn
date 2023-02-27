#!/bin/bash

# Spécifiez le nom du fichier Python que vous souhaitez lancer
python_file=experience/exp_class.py


# Vérifiez si le fichier existe
if [ -f "$python_file" ]; then
  for Re in 50000 100000 200000 500000 1000000
  do
    # Lancer le fichier Python
    echo "On lance l'experience pour Re = $Re"
    python3 "$python_file" 0 $Re
  done
else
  # Afficher un message d'erreur si le fichier n'existe pas
  echo "Le fichier $python_file n'existe pas"
fi