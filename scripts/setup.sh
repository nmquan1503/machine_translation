set -e

echo "Cloning Mamba..."
git clone https://github.com/nmquan1503/ssm-mamba.git ssm_mamba -q

cd ssm_mamba

echo "Installing Mamba..."
pip install . --no-build-isolation -q

cd ..

echo "Installing required libs..."
pip install sacrebleu -q

echo "Done."