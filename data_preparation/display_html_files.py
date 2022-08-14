import os
import sys

from bs4 import BeautifulSoup

file_list = sys.argv[1]
dir_name = os.path.dirname(file_list)
with open(file_list, 'r') as f, open(os.path.join(dir_name,"notes.txt"), 'w') as notes:
    stop = False
    for l in f.readlines():
        if not stop:
            html_file_name = l.strip()
            with open (os.path.join(dir_name, html_file_name)) as h:
                soup = BeautifulSoup(h)
                for s in soup.find_all('span'):
                    print(type(s), s)
                print(html_file_name)
            notes_to_add = input("enter notes : ")
            notes.write(f'{f} : {notes_to_add}\n')
            do_continue = "N"
            do_continue = input("continue? Y / N")
            stop = do_continue == "N"