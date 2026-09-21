"""Presigned PUT URLs for the merged model's files, one per file.

    python presign.py <name> files.json --bucket <bucket> [--region us-east-1] > urls.json

The Modal upload step PUTs each file to its URL, so Modal never holds AWS
credentials; the URLs are signed here with your local AWS profile and expire
after twelve hours.
"""

from __future__ import annotations

import argparse
import json
import sys

import boto3

# EXPIRES_S = 12 h: long enough for a 16 GB upload to finish on a slow day,
# short enough that a leaked urls.json goes stale the same day (convention).
EXPIRES_S = 12 * 3600


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name", help="model folder name in the bucket (the import's name)")
    ap.add_argument("files", help="files.json written by merge_upload.py::merge_entry")
    ap.add_argument("--bucket", required=True, help="S3 bucket the import job reads from")
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()

    files = json.load(open(args.files))["files"]
    s3 = boto3.client("s3", region_name=args.region)
    urls = {
        f["name"]: s3.generate_presigned_url(
            "put_object",
            Params={"Bucket": args.bucket, "Key": f"{args.name}/{f['name']}"},
            ExpiresIn=EXPIRES_S,
        )
        for f in files
    }
    json.dump(urls, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
