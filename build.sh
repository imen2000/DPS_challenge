set -o errexit

pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn
pip install flask 
pip install pickle 
pip install pandas