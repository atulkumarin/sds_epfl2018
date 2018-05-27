import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--synch', action='store_true')
parser.add_argument('--asynch', action='store_true')
args = parser.parse_args()

yaml_file_out = open('tmp.yaml', 'w')
yaml_file_in = None
with open('../kubernetes/config_template.yaml', 'r') as f:
    yaml_file_in = f.read()
config_json = None
with open('../config.json', 'r') as f:
    config_json = json.load(f)

if args.synch:
    config_json["ASYNCH"] = "0"
elif args.asynch:
    config_json["ASYNCH"] = "1"

config_json["nb_pods"] = int(config_json["nb_pods"])

yaml_file_out.write(yaml_file_in.format(**config_json))
yaml_file_out.close()

print(config_json["nb_pods"], end='')
