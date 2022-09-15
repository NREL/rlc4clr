import re

INCREASE_FACTOR = 3.5

with open('IEEE13busLineCodes.dss', 'r') as f:
    content = f.readlines()

new_content = []

for item in content:
    if 'rmatrix' in item or 'xmatrix' in item:
        old_val = re.findall("([\d.]+)", item)
        new_val = [str(round(float(x) * INCREASE_FACTOR, 4)) for x in old_val]

        for idx, o_val in enumerate(old_val):
            item = item.replace(o_val, new_val[idx])
    new_content.append(item)


with open('IEEE13busLineCodes.dss', 'w') as f:
    f.writelines(new_content)
