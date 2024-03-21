import os
import xml.etree.ElementTree as ET

# 定义文件夹路径
annotations_folder = r"F:\lubiao\VOCdevkit\VOC2007\Annotations"
jpeg_images_folder = r"F:\lubiao\VOCdevkit\VOC2007\JPEGImages"

# 获取Annotations文件夹下所有xml文件
xml_files = [file for file in os.listdir(annotations_folder) if file.endswith(".xml")]

# 遍历每个xml文件
for xml_file in xml_files:
    xml_path = os.path.join(annotations_folder, xml_file)

    # 解析XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 查找<name>标签的内容
    name_element = root.find(".//name")

    # 如果<name>标签的内容为"left"，则删除对应的xml文件和JPEGImages文件夹下的jpg文件
    if name_element is not None and name_element.text == "left":
        # 删除xml文件
        os.remove(xml_path)

        # 删除对应的jpg文件
        jpg_file = os.path.splitext(xml_file)[0] + ".jpg"
        jpg_path = os.path.join(jpeg_images_folder, jpg_file)
        if os.path.exists(jpg_path):
            os.remove(jpg_path)
