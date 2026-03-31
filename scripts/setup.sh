set -e

echo "Cloning Mamba..."
git clone https://github.com/state-spaces/mamba.git mamba -q

cd mamba

echo "Installing Mamba..."
pip install . --no-build-isolation -q

cd ..

echo "Installing required libs..."
pip install sacrebleu -q

echo "Done."