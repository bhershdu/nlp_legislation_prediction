import math
import os
from bs4 import BeautifulSoup
import sys
import re
import json

# NH Bill Specific features
# csE3E0FE0B highlighting css class
# csDD5E5F52 regular text css class
# need to remove tabs, (\d*), RSA\s\d*:\d*, roman numberals
# skip 'effective date'
# reject any bill with $

ANNO_DOMINI_KEYWORD = "In the Year"
AN_ACT_KEYWORD = "AN ACT"
BE_IT_ENACTED = "Be it enacted"
ROMAN_NUMERAL_REGEX = "[IVXLCDM]*\\."
MONEY_REGEX = "\\$"
HIGHLIGHT_CLASS = "csE3E0FE0B"
TEXT_CLASS = "csDD5E5F52"
EMPTY_TEXT = " "
EFFECTIVE_DATE = "Effective Date"
SECTION_KEYWORD = "Section"
BE_IT_ENACTED = "Be it Enacted"
SINGLE_TAB = "\t"
ALL_TABS = "(\\\\t)*"
NUMBER_FOLLOWED_BY_DOT = "(\\d)*\\."
NUMBER_FOLLOWED_BY_LETTER_PAREN = "(\\d)*[A-Za-c]*\\(\\.*\\)"
USC_KEYWORD = "U.S.C"
RSA_KEYWORD = "R.S.A"
NON_WORD_REGEX = "[^\\w]"
NUMBER_REGEX = "[\\d]*"
LETTER_TEXT_SEP_BY_HYPEN = "([A-Za-z0-9])*-([A-Za-z-09])*"
MULTIPLE_SPACES = "\\s+"
CONCURRENT_KEYWORD = "CONCURRENT"



file_list_file = sys.argv[1]
dir_name = os.path.dirname(file_list_file)


class FileState:

    def __init__(self, soup_instance: BeautifulSoup):
        self.soup_instance = soup_instance
        self.found_anno_domani = False
        self.found_an_act = False
        self.found_section = False
        self.found_enacted = False
        self.money_found = False
        self.text_sections = []
        self.lines_since_last_find = 0
        self.extract_threshold = 1

    def has_money(self, text):
        m = re.search(MONEY_REGEX, text)
        return m is not None

    def skip(self, tag):
        if tag["class"][0] == HIGHLIGHT_CLASS:
            return True
        elif len(tag.text) <= 1:
            return True
        elif SINGLE_TAB == tag.text:
            return True
        elif EFFECTIVE_DATE in tag.text:
            return True
        elif SECTION_KEYWORD in tag.text:
            return True
        elif BE_IT_ENACTED in tag.text:
            return True
        elif CONCURRENT_KEYWORD in tag.text:
            return True
        else:
            return False

    def process(self):
        spans = self.soup_instance.find_all('span')
        for s in spans:
            if not self.money_found:
                span_class = s["class"]
                span_text = s.text
                if ANNO_DOMINI_KEYWORD in s.text:
                    self.found_anno_domani = True
                    self.extract_threshold = 6
                    self.lines_since_last_find = 0

                if SECTION_KEYWORD in s.text:
                    self.found_section = True
                    self.extract_threshold = 1
                    self.lines_since_last_find = 0

                if AN_ACT_KEYWORD in s.text:
                    self.found_an_act = True
                    self.extract_threshold = 6
                    self.lines_since_last_find = 0

                if BE_IT_ENACTED in s.text:
                    self.found_enacted = True
                    self.extract_threshold = 3
                    self.lines_since_last_find = 0

                self.money_found = self.has_money(s.text)

                if self.found_anno_domani:
                    self.lines_since_last_find += 1
                    if self.lines_since_last_find >= self.extract_threshold:
                        if not self.skip(s):
                            self.text_sections.append(s.text)
            else:
                break
        if self.money_found is False:
            return self.text_sections
        else:
            return None


def clean(text_arr):
    # remove any tabs
    clean_text = " ".join(text_arr)
    clean_text = re.sub(ALL_TABS, "", clean_text)
    # remove roman numerals
    clean_text = re.sub(ROMAN_NUMERAL_REGEX, " ", clean_text)
    # remove numbers
    clean_text = re.sub(NUMBER_FOLLOWED_BY_DOT, "", clean_text)
    # numbers followed by letters and parenthese
    clean_text = re.sub(NUMBER_FOLLOWED_BY_LETTER_PAREN, "", clean_text)
    # remove law type refernce
    clean_text = clean_text.replace(USC_KEYWORD, "")
    clean_text = clean_text.replace(RSA_KEYWORD, "")
    clean_text = re.sub(LETTER_TEXT_SEP_BY_HYPEN, "", clean_text)
    clean_text = re.sub(NUMBER_REGEX, "", clean_text)
    clean_text = re.sub(NON_WORD_REGEX, " ", clean_text)
    clean_text = re.sub(MULTIPLE_SPACES, " ", clean_text)
    return clean_text.strip().lower()


def extract_text(html_file_name):
    expected_output_name = html_file_name.replace(".html", ".json")
    expected_bill_file = html_file_name.replace("_text","").replace(".html",".json")

    if os.path.exists(os.path.join(dir_name, expected_output_name)):
        print(f"output file {expected_output_name} already exists. skipping...")
    else:
        with open(os.path.join(dir_name, html_file_name)) as h:
            soup = BeautifulSoup(h, features="lxml")
            # get the set of span tags
            processor = FileState(soup)
            text_sections = processor.process()
            if text_sections is not None:
                # clean
                clean_text = clean(text_sections)
                # read bill file, add text, write to new file
                with open(os.path.join(dir_name, expected_bill_file), 'r') as b, \
                        open(os.path.join(dir_name, expected_output_name), 'w') as out_file:
                    data = json.load(b)
                    data["bill"]["full_text"] = clean_text
                    json.dump(data, out_file, indent=4)


def main():
    with open(file_list_file, 'r') as f:
        for line in f:
            html_file = line.strip()
            extract_text(html_file)


if __name__ == "__main__":
    main()
