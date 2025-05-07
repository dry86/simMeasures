import json
import os


def compare_dicts(a, b, prefix=""):
    only_in_a = {}
    only_in_b = {}
    different = {}
    same = {}

    keys_a = set(a.keys())
    keys_b = set(b.keys())

    for key in keys_a - keys_b:
        only_in_a[key] = a[key]
    for key in keys_b - keys_a:
        only_in_b[key] = b[key]
    for key in keys_a & keys_b:
        if isinstance(a[key], dict) and isinstance(b[key], dict):
            sub = compare_dicts(a[key], b[key], prefix + key + ".")
            if sub["only_in_A"] or sub["only_in_B"] or sub["different"]:
                different[key] = sub
            else:
                same[key] = a[key]
        elif a[key] != b[key]:
            different[key] = {"A": a[key], "B": b[key]}
        else:
            same[key] = a[key]
    return {
        "only_in_A": only_in_a,
        "only_in_B": only_in_b,
        "different": different,
        "same": same
    }

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    path_a = 'simMeasures/simConfig/config-Coder.json'
    path_b = 'simMeasures/simConfig/config-VL.json'
    if not (os.path.exists(path_a) and os.path.exists(path_b)):
        print('配置文件路径不存在')
        return
    config_a = load_json(path_a)
    config_b = load_json(path_b)
    result = compare_dicts(config_a, config_b)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main() 