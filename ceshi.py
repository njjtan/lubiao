import os
import xml_obj.etree.ElementTree as ET

def update_xml_file(file_path, old_value, new_value):
    tree = ET.parse(file_path)
    root = tree.getroot()

    found = False
    for obj in root.findall('.//object/name'):
        if obj.text == old_value:
            obj.text = new_value
            found = True

    if found:
        tree.write(file_path)

def batch_update_folder(folder_path, old_value, new_value):
    for filename in os.listdir(folder_path):
        if filename.endswith(".xml"):
            file_path = os.path.join(folder_path, filename)
            update_xml_file(file_path, old_value, new_value)

# 替换 Positive needle 为 Positive-needle
batch_update_folder('F:\lubiao\VOCdevkit\VOC2007\Annotations', 'pointelle', 'Pointelle')
