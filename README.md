

pip install uv

uv init testinvoke4/
uv add invoke
uv add scikit-learn
uv add scikit-multilearn-ng
uv add iterative-stratification

source .venv/bin/activate

invoke -l
