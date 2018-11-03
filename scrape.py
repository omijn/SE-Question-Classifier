import requests
from bs4 import BeautifulSoup
import argparse
import json
import time
import subprocess


class SEApiClient:
    base_url = "https://api.stackexchange.com/2.2/"
    sites_endpoint = "sites"
    tags_endpoint = "tags"
    questions_endpoint = "questions"

    @classmethod
    def sites_api(cls):
        r = requests.get(cls.base_url + cls.sites_endpoint, params={"pagesize": 500})
        response = r.json()
        formatted_response = {"items": {}}
        for site_info in response["items"]:
            if site_info["site_type"] == "meta_site":
                continue
            formatted_response["items"][site_info["api_site_parameter"]] = {
                "name": site_info["name"],
                "site_url": site_info["site_url"],
                "high_resolution_icon_url": site_info["high_resolution_icon_url"],
                "logo_url": site_info["logo_url"],
                "site_type": site_info["site_type"]
            }
        return formatted_response

    @classmethod
    def tags_api(cls, tagcount, site_source):
        formatted_response = {"items": {}}
        print("Currently getting tags for: ")
        for site in site_source["items"].keys():
            print(site)
            r = requests.get(cls.base_url + cls.tags_endpoint,
                             params={"site": site, "pagesize": tagcount, "order": "desc", "sort": "popular"})
            response = r.json()
            formatted_response["items"][site] = list(map(lambda item: item["name"], response["items"]))
            time.sleep(1)
        return formatted_response

    @classmethod
    def questions_api(cls, site_source, tag_source, question_limit, questions_metafile):
        # TODO: do something if file doesn't exist
        completed_sites_file = open("completed_sites", "r+")
        completed_sites = completed_sites_file.read().split("\n")
        try:
            completed_sites.remove("")
        except ValueError:
            pass

        with open(questions_metafile, "r+") as qfile:
            qfile.seek(0)
            try:
                qdata = json.load(qfile)
            except json.decoder.JSONDecodeError:
                qdata = {}

            for site in site_source["items"]:
                if site in completed_sites:
                    continue
                if site not in qdata:
                    qdata[site] = {"questions": {}, "completedtags": {}}

                current_site_questions_dict = qdata[site]["questions"]
                current_site_completedtags_dict = qdata[site]["completedtags"]
                for tag in tag_source["items"][site]:
                    print("\r{}: {}".format(site, tag), end='')
                    if tag in current_site_completedtags_dict.keys():
                        continue

                    page = 0
                    current_question_count = 0
                    while current_question_count < question_limit:  # if we're unlucky this loop will run more than once
                        page += 1
                        time.sleep(1)

                        r = requests.get(cls.base_url + cls.questions_endpoint,
                                         params={"site": site, "tagged": tag, "page": page,
                                                 "pagesize": question_limit + 20, "order": "desc", "sort": "votes"})
                        response = r.json()

                        try:
                            response_items = response["items"]
                        except KeyError:
                            print("\rError {} ({}): {}".format(response["error_id"], response["error_name"],
                                                               response["error_message"]))
                            return
                            # print("[1/5] Killing old VPN process if it exists.")
                            # kill_openconnect = subprocess.run(args='sudo -S killall -s 9 -v openconnect'.split(),
                            #                                   input=open("sudopass", "r").read(), encoding="utf8", stdout=subprocess.PIPE,
                            #                                   stderr=subprocess.PIPE)
                            # print("[2/5] ", end='')
                            # print(kill_openconnect.stdout.strip("\n"), end='')
                            # print(kill_openconnect.stderr.strip("\n"), end='')
                            # print("[3/5] Creating new VPN process.")
                            # proc = subprocess.Popen(args=open("command", "r").read().split(), encoding="utf8", stdin=subprocess.PIPE,
                            #                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            # try:
                            #     print("[4/5] Sending sudo and USC VPN passwords to process STDIN.")
                            #     o, e = proc.communicate(input=open("pass", "r").read(), timeout=15)
                            # except subprocess.TimeoutExpired:
                            #     proc.stdin.close()
                            #     proc.stdout.close()
                            #     print(o)
                            #     print(e)
                            #     print("[5/5] Successfully created new VPN process and closed I/O stream.")

                        for question_info in response_items:
                            question_id = question_info["question_id"]
                            if str(question_id) in map(str, qdata[site][
                                "questions"].keys()):  # don't repick questions that repeat across tags
                                continue

                            current_site_questions_dict[question_id] = {"tag": tag, "link": question_info["link"],
                                                                        "title": question_info["title"]}
                            current_question_count += 1
                            if current_question_count == question_limit:
                                break
                        if page == 2:  # give up after looking for two pages of questions and move on to the next tag
                            break
                    current_site_completedtags_dict[tag] = 1
                    qfile.seek(0)
                    qfile.write(json.dumps(qdata))

                qfile.seek(0)
                qfile.write(json.dumps(qdata))

                completed_sites_file.write(site + "\n")

            qdata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites-api", "--sa", action="store_true")
    parser.add_argument("--sites-outfile", "--so")
    parser.add_argument("--sites-infile", "--si")

    parser.add_argument("--tags-api", "--ta", action="store_true")
    parser.add_argument("--tags-outfile", "--to")
    parser.add_argument("--tags-infile", "--ti")

    parser.add_argument("--questions-api", "--qa", action="store_true")
    parser.add_argument("--questions-metafile", "--qf")

    parser.add_argument("--tagcount", "-c", type=int, default=70)
    parser.add_argument("--questioncount", "-q", type=int, default=30)

    args = parser.parse_args()

    if args.sites_api:
        site_source = SEApiClient.sites_api()
        with open(args.sites_outfile, "w") as f:
            json.dump(site_source, f)
    else:
        with open(args.sites_infile, "r") as f:
            site_source = json.load(f)

    if args.tags_api:
        tag_source = SEApiClient.tags_api(args.tagcount, site_source)
        with open(args.tags_outfile, "w") as f:
            json.dump(tag_source, f)
    else:
        with open(args.tags_infile, "r") as f:
            tag_source = json.load(f)

    if args.questions_api:
        SEApiClient.questions_api(site_source, tag_source, args.questioncount,
                                  args.questions_metafile)

    with open(args.questions_metafile, "r") as f:
        question_source = json.load(f)


if __name__ == '__main__':
    main()
