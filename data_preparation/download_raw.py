import base64
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
    """
    Return the api key from the secrets file
    :return: the api key value
    """
    key_file = os.path.join(os.getcwd(), "..", "secrets","legal_scan_key.txt")
    with open(key_file, 'r') as f:
        api_key = f.readline()
    return api_key


def get_current_session(api_key, offset=0):
    """
    Use the rest api to get the list of session that match the current minus offset year and the previous year.
    Writes the results to file.
    :param api_key: the api key
    :param offset: a year offset value (0-n)
    :return: a json object of the matching session information
    """
    curr_year = datetime.date.today().year-offset
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
    """
    Use the rest api to get session-based list of legislators.
    Writes the results to a file
    :param session_id: the session id to get the list of legislators
    :param api_key: the api key
    :return:
    """
    response = requests.get(API_LEGISCAN_COM, params={"key": api_key, "op": "getSessionPeople", "id": session_id})
    output_path = os.path.join(os.getcwd(), "../data/raw", f'people_{session_id}.json')
    with open(output_path, 'w') as f:
        json.dump(response.json(), f, indent=4)


def get_first_bill(status_value, session_id, api_key):
    """
    Get the first bill for a given status value and session id.
    Write each bill to it's own file.
    Used in initial use of the API to prevent rate limit issues.
    :param status_value: the status value to look for
    :param session_id: the session id
    :param api_key: the api key
    :return: None
    """
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
                                json.dump(response.json(), f, indent=4)
                            done = True

def get_bill_text(session_id, bill_id, bill_json, api_key):
    """
    Using the legiscan API and general HTML requests, get the full text of each bill.
    Write the results to file
    :param session_id: the session id
    :param bill_id: the bill id
    :param bill_json: the parent bill json
    :param api_key: the api key
    :return: None
    """
    bill_data = bill_json["bill"]
    if "texts" in bill_data:
        texts_arr = bill_data["texts"]
        if len(texts_arr) > 0:
            # pick the last one
            text_doc = texts_arr[-1]
            doc_id = text_doc["doc_id"]
            mime_type = text_doc["mime"]
            if mime_type == "text/html":
                file_ext = "html"
            else:
                print("unhandled mime type ", mime_type)
                file_ext = "txt"
            response = requests.get(API_LEGISCAN_COM,
                                    params={"key": api_key, "op": "getBillText", "id": doc_id})
            if response.status_code == 200:
                text_doc = response.json()
                base64_doc = text_doc["text"]["doc"]
                doc_text = str(base64.decodebytes(bytes(base64_doc, 'utf-8')))
                # print(doc_text)
                text_file_path = os.path.join(os.getcwd(), "../data/raw", f'bill_text_{session_id}_{bill_id}.{file_ext}')
                with open(text_file_path, 'w') as f:
                    f.write(doc_text)

def get_all_bills(session_id, api_key):
    """
    Get all bills for a session id
    :param session_id: the session id
    :param api_key: the api key
    :return: None
    """
    print(f'downloading all bills for session {session_id}')
    master_file_path = os.path.join(os.getcwd(), "..", "data", "raw", f'master_list_{session_id}.json')
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
                    get_bill_text(session_id, b["bill_id"], response.json(), api_key)
                    with open(bill_file_path, 'w') as f:
                        json.dump(response.json(),f, indent=4)
                else:
                    print(f'file for bill {b["bill_id"]} exists. skipping')


def get_bill_list(session_id, api_key):
    """
    Use Search API to get list of bills in a sessions. Only makes API calls if the output file does not exist.
    :param session_id: the session id
    :param api_key: the api key
    :return: None
    """
    output_path = os.path.join(os.getcwd(), "../data/raw", f'master_list_{session_id}.json')
    if not os.path.exists(output_path):
        response = requests.get(API_LEGISCAN_COM,
                                params={"key": api_key, "op": "getMasterList", "id": session_id})
        with open(output_path, 'w') as f:
            json.dump(response.json(), f, indent=4)
    else:
        print(f'{output_path} already exists. skipping.')


def main():
    """
    The main function
    :return: None
    """
    api_key = load_api_key()
    session = get_current_session(api_key,2)
    for s in session["sessions"]:
        get_bill_list(s["session_id"], api_key)
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