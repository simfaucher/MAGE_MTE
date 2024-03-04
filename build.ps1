echo "Installing requirements"
pip install -r requirements.txt --quiet
echo "Installing build requirements"
pip install -r build-requirements.txt --quiet
echo "Installing cx_Freeze Beta version as version for python 3.12 is not available at the time of writing this script."
pip install --force --quiet --no-cache --pre --extra-index-url https://marcelotduarte.github.io/packages/ cx_Freeze
python setup.py build
echo "Build complete"