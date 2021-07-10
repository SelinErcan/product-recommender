# Related Products Recommendation

#### This repo includes

1. EDA notebook (Exploratory Data Analysis)
2. Recommender model
3. Recommender API
4. Postman collection example for the API (change URL if ngrok is used)

## How to run

```python

sudo apt update
sudo apt install python3-dev python3-pip

python3 -m venv ./venv
source ./venv/bin/activate
pip install -U pip

pip install -r requirements.txt -U

python3 train.py  # to train model
python3 api.py    # to run api
./ngrok http 5000 # (optional) to run ngrok tunnel to create a public URL for the api 

```



