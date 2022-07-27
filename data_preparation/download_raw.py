import os.path
import sys
import json
import datetime
import requests

API_LEGISCAN_COM = "https://api.legiscan.com"

"""
Download a file from the nh senate site
"""
legal_scan_url = 'https://api.legalscan.com/'


def load_api_key():
    key_file = os.path.join(os.getcwd(), "..", "secrets","legal_scan_key.txt")
    with open(key_file, 'r') as f:
        api_key = f.readline()
    return api_key


def get_current_session(api_key):
    curr_year = datetime.date.today().year
    prev_year = curr_year-1
    print(os.getcwd())
    response = requests.get(API_LEGISCAN_COM,
                            params={'key': api_key, "op": "getSessionList", "state": "NH"})

    sessions_json = response.json()
    sessions = sessions_json["sessions"]
    session_node_list = list(filter(lambda node: node["year_start"] in [prev_year,curr_year], sessions))
    sessions = {"sessions": session_node_list}
    output_path = os.path.join(os.getcwd(),"../data/raw",f'sessions_{prev_year}_{curr_year}_session_data.json')
    with open(output_path, 'w') as f:
        json.dump(sessions, f, indent=4)
    return sessions


def get_legislators(session_id, api_key):
    response = requests.get(API_LEGISCAN_COM, params={"key": api_key, "op": "getSessionPeople", "id": session_id})
    output_path = os.path.join(os.getcwd(), "../data/raw", f'people_{session_id}.json')
    with open(output_path, 'w') as f:
        json.dump(response.json(), f, indent=4)


def get_first_bill(status_value, session_id, api_key):
    master_file_path = os.path.join(os.getcwd(), "../data/raw", f'master_list_{session_id}.json')
    done = False
    with open(master_file_path, 'r') as m:
        bill_list = json.load(m)
        for k in bill_list["masterlist"]:
            if not done:
                b = bill_list["masterlist"][k]
                if "status" in b:
                    if b["status"] == status_value:
                        bill_file_path = os.path.join(os.getcwd(),"../data/raw", f'bill_{session_id}_{b["bill_id"]}.json')
                        if not os.path.exists(bill_file_path):
                            response = requests.get(API_LEGISCAN_COM,
                                                    params={"key": api_key, "op": "getBill", "id": b["bill_id"]})
                            with open(bill_file_path, 'w') as f:
                                json.dump(response.json(),f, indent=4)
                            done = True


def get_all_bills(session_id, api_key):
    print(f'downloading all bills for session {session_id}')
    master_file_path = os.path.join(os.getcwd(), "../data/raw", f'master_list_{session_id}.json')
    with open(master_file_path, 'r') as m:
        bill_list = json.load(m)
        for k in bill_list["masterlist"]:
            b = bill_list["masterlist"][k]
            if "status" in b:
                bill_file_path = os.path.join(os.getcwd(),"../data/raw", f'bill_{session_id}_{b["bill_id"]}.json')
                if not os.path.exists(bill_file_path):
                    print(f'fetching bill {b["bill_id"]}')
                    response = requests.get(API_LEGISCAN_COM,
                                            params={"key": api_key, "op": "getBill", "id": b["bill_id"]})
                    with open(bill_file_path, 'w') as f:
                        json.dump(response.json(),f, indent=4)
                else:
                    print(f'file for bill {b["bill_id"]} exists. skipping')


def get_bill_list(session_id, api_key):
    output_path = os.path.join(os.getcwd(), "../data/raw", f'master_list_{session_id}.json')
    if not os.path.exists(output_path):
        response = requests.get(API_LEGISCAN_COM,
                                params={"key": api_key, "op": "getMasterList", "id": session_id})
        with open(output_path, 'w') as f:
            json.dump(response.json(), f, indent=4)
    else:
        print(f'{output_path} already exists. skipping.')


def main():
    api_key = load_api_key()
    session = get_current_session(api_key)
    for s in session["sessions"]:
        get_legislators(s["session_id"], api_key)
        get_all_bills(s["session_id"], api_key)
#        get_all_bills(0, s["session_id"], api_key)
#        get_first_bill(0, s["session_id"], api_key)
#        get_first_bill(1, s["session_id"], api_key)
#        get_first_bill(2, s["session_id"], api_key)
#        get_first_bill(3, s["session_id"], api_key)
#        get_first_bill(4, s["session_id"], api_key)

if __name__=="__main__":
    main()