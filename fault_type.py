import re
import argparse

def update_variable(file_path, variable_name, new_value):

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = r'({}\s*=\s*)\d+'.format(re.escape(variable_name))
    new_content = re.sub(pattern, r'\g<1>{}'.format(new_value), content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"Updated the value of {variable_name} in {file_path} to {new_value}")

def main():
    parser = argparse.ArgumentParser(description="Update the type of fault injection")
    parser.add_argument("option", choices=["rd", "cur", "mix", "none"],
                        help="Select fault types: rd, cur, mix, none")
    args = parser.parse_args()

    if args.option == "rd":
        update_variable("selfdrive/modeld/modeld.py", "FI_cur", 0)
        update_variable("selfdrive/controls/radard.py", "FI_relDis", 1)
    elif args.option == "cur":
        update_variable("selfdrive/modeld/modeld.py", "FI_cur", 2)
        update_variable("selfdrive/controls/radard.py", "FI_relDis", 0)
    elif args.option == "mix":
        update_variable("selfdrive/modeld/modeld.py", "FI_cur", 1)
        update_variable("selfdrive/controls/radard.py", "FI_relDis", 1)
    elif args.option == "none":
        update_variable("selfdrive/modeld/modeld.py", "FI_cur", 0)
        update_variable("selfdrive/controls/radard.py", "FI_relDis", 0)

if __name__ == "__main__":
    main()
