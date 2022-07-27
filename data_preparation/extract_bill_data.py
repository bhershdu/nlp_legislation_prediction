import os
import fnmatch
import sys
import json

party_labels = ["D","R", "I"]

def main():
    path_to_scan = sys.argv[1]
    output_path = sys.argv[2]

    people_party = {}
    for root, dirs, files in os.walk(path_to_scan):
        for f in files:
            if fnmatch.fnmatch(f, 'people*'):
                with open(os.path.join(root, f), 'r') as p_file:
                    p_data = json.load(p_file)
                    for p in p_data['sessionpeople']['people']:
                        people_party[p['people_id']] = p['party']

    for root, dirs, files in os.walk(path_to_scan):
        for f in files:
            if fnmatch.fnmatch(f, "bill*.json"):
                output_file_path = os.path.join(output_path, f'summary_{f}')
                print(f'processing {f}')
                if not os.path.exists(output_file_path):
                    with open(output_file_path, 'w') as output_file, open(os.path.join(root, f), 'r') as input_file:
                        data = json.load(input_file)
                        output_data = {}
                        output_data['text'] = data['bill']["title"]
                        # have seen issues at work where a label value of 0 can cause problems during training,
                        # so we increment the status by one
                        output_data['status'] = data['bill']['status']+1
                        party_sponsor = []
                        missing = False
                        for s in data['bill']['sponsors']:
                            if s['people_id'] in people_party:
                                party_sponsor.append(people_party[s['people_id']])
                            else:
                                missing = True
                        if not missing:
                            unique_parties = set(party_sponsor)
                            if len(unique_parties) == 1:
                                party_id = party_labels.index(list(unique_parties)[0])
                            elif len(unique_parties) > 1:
                                party_id = len(party_labels)
                        else:
                            party_id = None
                        output_data['party'] = party_id
                        json.dump(output_data, output_file)
                else:
                    print(f'{output_file_path} already exists')


if __name__ == "__main__":
    main()