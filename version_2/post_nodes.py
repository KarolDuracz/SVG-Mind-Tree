# pip install requests
# python post_nodes.py
#
# Quickstart - how to get started with Requests - https://requests.readthedocs.io/en/latest/user/quickstart/
#

import requests

url = "http://localhost:5000/api/nodes"
headers = {"Content-Type": "application/json"}

start_value = 103           # change start ID node
loops = 20                 # number of nodes

for i in range(loops):
    connect_to = start_value + i
    payload = {
        "name": f"#{connect_to} My Node",
        "connect_to": connect_to,
        "sequence": "true"
    }
    response = requests.post(url, headers=headers, json=payload)
    print(f"Request {i+1}: connect_to={connect_to}, status={response.status_code}, response={response.text}")
