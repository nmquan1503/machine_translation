set -e

echo "Cloning Mamba..."
git clone https://github.com/state-spaces/mamba.git mamba -q
export PYTHONPATH="$PWD/mamba:$PYTHONPATH"
touch mamba/__init__.py

cd mamba

echo "Installing Mamba..."
pip install . --no-build-isolation -q

cd ..

echo "Installing Causal-conv1d..."
git clone https://github.com/Dao-AILab/causal-conv1d.git -q
cd causal-conv1d
pip install . --no-build-isolation -q
cd ..

echo "Installing required libs..."
pip install sacrebleu -q

echo "Done."