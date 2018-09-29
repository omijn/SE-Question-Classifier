import requests
from bs4 import BeautifulSoup
import argparse
import json


class SEApiClient:
    base_url = "https://api.stackexchange.com/2.2/"
    sites_endpoint = "sites"
    tags_endpoint = "tags"

    def sites_api(self):
        r = requests.get(self.base_url + self.sites_endpoint, params={"pagesize": 300})
        response = r.json()
        formatted_response = {"items": {}}
        for site_info in response["items"]:
            formatted_response["items"][site_info["api_site_parameter"]] = {
                "name": site_info["name"],
                "site_url": site_info["site_url"],
                "high_resolution_icon_url": site_info["high_resolution_icon_url"],
                "logo_url": site_info["logo_url"],
                "site_type": site_info["site_type"]
            }
        return formatted_response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites-api", "--sa", action="store_true")
    parser.add_argument("--sites-outfile", "--so")
    parser.add_argument("--sites-infile", "--si")

    parser.add_argument("--tags-api", "--ta", action="store_true")
    parser.add_argument("--tags-outfile", "--to")
    parser.add_argument("--tags-infile", "--ti")

    parser.add_argument("--tagcount", "-c", type=int, default=70)
    parser.add_argument("--questioncount", "-q", type=int, default=30)

    args = parser.parse_args()

    if args.sites_api:
        site_source = SEApiClient().sites_api()
        with open(args.sites_outfile, "w") as f:
            json.dump(site_source, f)
    else:
        with open(args.sites_infile, "r") as f:
            site_source = json.load(f)


if __name__ == '__main__':
    main()
