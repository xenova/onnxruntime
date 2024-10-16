import os
import json
import argparse
import time

parser = argparse.ArgumentParser("Upload and run BrowserStack tests")

parser.add_argument("--token", type=str, help="BrowserStack user ID")
parser.add_argument("--app_apk_path", type=str, help="Path to the app APK")
parser.add_argument("--test_apk_path", type=str, help="Path to the test suite APK")
# TODO: add link to browserstack documentation of available device strings you can pass in
parser.add_argument("--devices", type=str, nargs="+", help="List of devices to run the tests on")

args = parser.parse_args()

def curl_then_parse_json(curl_command):
    print("this is what is being passed to os.popen!")
    print()
    print(curl_command)
    print()
    response_str = os.popen(curl_command).read()
    print("Response string:", response_str)

    if len(response_str) == 0:
        raise Exception("No response from BrowserStack")
    try:
        return json.loads(response_str)
    except:
        raise Exception("Invalid JSON response from BrowserStack")

def upload_apk_parse_json(post_url, apk_path):
    upload_command = "curl -u \"{token}\" -X POST \"{post_url}\" -F \"file=@{apk_path}\"".format(token=args.token, post_url=post_url, apk_path=apk_path)
    return curl_then_parse_json(upload_command)

upload_app_json = upload_apk_parse_json("https://api-cloud.browserstack.com/app-automate/espresso/v2/app", args.app_apk_path)
upload_test_json = upload_apk_parse_json("https://api-cloud.browserstack.com/app-automate/espresso/v2/test-suite", args.test_apk_path)

# time.sleep(3)

# TODO: make the list of devices also an option
# build_command_unformatted = """
# curl -u "{token}" \\
# -X POST "https://api-cloud.browserstack.com/app-automate/espresso/v2/build" \\
# -d '{{"devices": ["Samsung Galaxy S22-12.0"], "app": "{app_url}", "testSuite": "{test_app_url}"}}' \\
# -H "Content-Type: application/json"
# """

build_command_unformatted = "curl -u \"{token}\" -X POST \"https://api-cloud.browserstack.com/app-automate/espresso/v2/build\" -d '{{\"devices\": [\"Samsung Galaxy S22-12.0\"], \"app\": \"{app_url}\", \"testSuite\": \"{test_app_url}\" }}' -H \"Content-Type: application/json\""

build_command = build_command_unformatted.format(token=args.token, app_url=upload_app_json["app_url"], test_app_url=upload_test_json["test_suite_url"])

build_reponse_json = curl_then_parse_json(build_command)

build_response_str = json.dumps(build_reponse_json, indent=2)
print("Build response:", build_response_str)
