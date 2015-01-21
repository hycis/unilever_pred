#!/usr/bin/env python

import argparse
import json
import glob
import os
import requests
import urlparse

from bs4 import BeautifulSoup


def main(args):
    with open(args.opt) as f:
        opts = json.load(f)

    url = opts['url']
    parsed = urlparse.urlparse(url)

    s = requests.Session()
    s.headers.update({
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "en-US,en;q=0.8",
        "Host": parsed.netloc,
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
        "Cookie": args.cookie,
        "DNT": 1,
    })

    r = s.get(url)
    doc = BeautifulSoup(r.text)
    form = doc.find('form', class_='dropzone')
    data = {(input['name'] or input['id']): (input['value'] or '') for input in form.find_all('input', type='hidden')}
    data['challengeEM'] = opts['challengeEM']
    data['userId'] = args.user
    data['userSubmissionId'] = args.sub
    post_url = urlparse.urljoin(url, form['action'])

    print(data)
    print(post_url)
    print("")

    count = 1
    for path in glob.glob(args.src):
        #post_url = 'http://httpbin.org/post'
        print("{}: Uploading {} ...".format(count, path))
        count += 1
        in_filename = os.path.basename(path)
        in_file = open(path, "r")
        files = {'file': (in_filename, in_file, "text/csv")}
        headers = {
            "X-Requested-With": "XMLHttpRequest",
            "X-File-Name": in_filename,
            "Cache-Control": "no-cache",
            "Accept": "application/json",
            "Referer": url,
            "Origin": parsed.scheme + "://" + parsed.netloc,
        }
        r = s.post(post_url, files=files, data=data, headers=headers)
        if "New personal best submission is uploaded" in r.text:
            print("\tBest {} ...".format(path))
        in_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='submit results.')
    parser.add_argument("-s", "--src", help="input dir path (glob)", required=True)
    parser.add_argument("-c", "--cookie", help="login cookie", required=True)
    parser.add_argument("-u", "--user", help="user id", required=True)
    parser.add_argument("--sub", help="submission id", required=True)
    parser.add_argument("-o", "--opt", help="option file", required=True)
    args = parser.parse_args()
    main(args)
