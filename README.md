
 # Installation de uv:

pip install uv

# Installation de l'environnement
uv init serge-multilearn/
cd serge-multilearn/
uv add invoke
uv add scikit-learn
uv add scikit-multilearn-ng
uv add iterative-stratification

# initalisation de l'environnement
source .venv/bin/activate

# utilisation de invoke
invoke -l
invoke run-all

# les fichiers de confi
config.py

# les résultats
best_hyperparameters.json
resultats.txt
