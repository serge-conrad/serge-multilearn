

# SUR GRID 5000
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env (or restart shell)
cd serge-multilearn/
uv add scikit-learn

sconrad@flyon:~/serge-multilearn$ uv add scikit-learn
Using CPython 3.11.10
Creating virtual environment at: .venv

oarsub -I
source .venv/bin/activate
invoke -l 
invoke run-all

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
