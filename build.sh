set -o errexit
/opt/render/project/src/.venv/bin/python -m pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn
pip install flask
pip install pandas
pip install matplotlib 
pip install scikit-learn

