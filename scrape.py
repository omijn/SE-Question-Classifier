import time
import json
import requests
from bs4 import BeautifulSoup


def scrape(html):
    soup = BeautifulSoup(html, 'html.parser')
    # qtitle = soup.find("a", attrs={"class": "question-hyperlink"}).string
    try:
        qcontent = soup.find("div", attrs={"class": "post-text"}).text
    except:
        qcontent = "FAILED"
    return qcontent


def main():
    metadata_fp = open("qmeta.json")
    completed_fp = open("completed_questions.json", "r+")
    failed_fp = open("failed.txt", "a")

    # load question metadata (file with question id, link, title and tag)
    qmeta = json.load(metadata_fp)
    metadata_fp.close()

    # load file that saves progress of questions scraped so far
    try:
        qcompleted = json.load(completed_fp)
    except json.decoder.JSONDecodeError:  # if file is empty
        qcompleted = {"completed_sites": []}

    completed_sites = qcompleted["completed_sites"]


    for sitename, siteqmetadata in qmeta.items():
        if sitename in completed_sites or sitename.endswith(".meta") or sitename.startswith("meta"):
            continue

        # load file that stores the actual question text
        try:
            num_completed_sites = len(qcompleted["completed_sites"])
            if num_completed_sites <= 72:
                filenum = ""
            elif num_completed_sites <= 100:
                filenum = 1
            elif num_completed_sites <= 130:
                filenum = 2
            elif num_completed_sites <= 160:
                filenum = 3
            else:
                filenum = 4

            questions_filename = "qdata" + str(filenum) + ".json"

            questions_fp = open(questions_filename, "r+")
            qdata = json.load(questions_fp)
        except json.decoder.JSONDecodeError:  # if file is empty
            qdata = {}

        try:
            qcompleted[sitename]
        except KeyError:
            qcompleted[sitename] = []

        print("Writing to " + questions_filename)

        completed_questions_for_this_site = qcompleted[sitename]
        failcount = 0
        for qid, qmetadata in siteqmetadata['questions'].items():
            if qid in completed_questions_for_this_site:
                continue
            qurl = qmetadata['link']
            r = requests.get(qurl)
            qcontent = scrape(r.text)
            if r.status_code != 200 or qcontent == "FAILED":
                failed_fp.write(qurl + "\n")
                failcount += 1
                if failcount >= 5:
                    print("Five consecutive failures occurred. Something's wrong!")
                    return
                continue
            failcount = 0
            qtitle = qmetadata['title']
            qtag = qmetadata['tag']

            try:
                qdata[sitename]
            except KeyError:
                qdata[sitename] = []

            qdata[sitename].append({
                qid: {
                    "title": qtitle,
                    "content": qcontent,
                    "tag": qtag
                }
            })
            questions_fp.seek(0)
            json.dump(qdata, questions_fp)

            qcompleted[sitename].append(qid)
            completed_fp.seek(0)
            json.dump(qcompleted, completed_fp)
            time.sleep(0.35)
        qcompleted["completed_sites"].append(sitename)
        completed_fp.seek(0)
        json.dump(qcompleted, completed_fp)
        questions_fp.close()

    completed_fp.close()


if __name__ == '__main__':
    main()