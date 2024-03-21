import os

# # 图片文件夹路径
# image_folder = 'F:\lubiao\VOCdevkit\VOC2007\JPEGImages'
#
# # XML文件夹路径
# xml_folder = 'F:\lubiao\VOCdevkit\VOC2007\Annotations'
#
# # 获取图片和XML文件列表
# image_files = os.listdir(image_folder)
# xml_files = os.listdir(xml_folder)
#
# # 遍历图片文件夹，删除没有对应XML的图片文件
# for image_file in image_files:
#     # 获取图片文件名（不包括扩展名）
#     image_filename = os.path.splitext(image_file)[0]
#
#     # 构建对应的XML文件名
#     corresponding_xml = image_filename + '.xml'
#
#     # 如果对应的XML文件不存在，则删除当前图片文件
#     if corresponding_xml not in xml_files:
#         image_path = os.path.join(image_folder, image_file)
#         os.remove(image_path)
#         print(f"Deleted: {image_path}")
#
# print("Task completed.")

import os

def rename_images(input_folder):
    # 获取文件夹中的所有文件
    files = os.listdir(input_folder)

    # 确保文件夹路径以斜杠结尾
    if not input_folder.endswith('/'):
        input_folder += '/'

    # 初始化计数器
    count = 1

    # 遍历文件夹中的所有文件
    for file in files:
        # 检查文件是否是图片文件（你可以根据实际需要扩展这个检查）
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.xml')):
            # 构建新的文件名
            new_name = f"{count}.{file.split('.')[-1]}"

            # 构建完整的文件路径
            old_path = f"{input_folder}{file}"
            new_path = f"{input_folder}{new_name}"

            # 处理文件名冲突
            while os.path.exists(new_path):
                count += 1
                new_name = f"{count}.{file.split('.')[-1]}"
                new_path = f"{input_folder}{new_name}"

            # 重命名文件
            os.rename(old_path, new_path)

            # 更新计数器
            count += 1

# 指定图片文件夹的路径
input_folder = 'F:\lubiao\VOCdevkit\VOC2007\JPEGImages'

#调用函数进行重命名
rename_images(input_folder)

