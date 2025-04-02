from PIL import Image, ImageDraw, ImageFont
import textwrap
from translate import Translator

def translate_text_english(text):
    translator = Translator(from_lang="zh", to_lang="en")
    return translator.translate(text) if translator else text

def process_images_conclusion_english(images_conclusion):
    image_map = {img: part for part, imgs in images_conclusion.items() for img in imgs}
    positions = [(x, y) for y in [110, 310] for x in [50, 260, 470, 680]][:len(image_map)]
    return image_map, positions

def wrap_text_english(text, width=80):
    return textwrap.wrap(text, width=width)

def create_report_english(region_descriptions, conclusion, images_conclusion, part_names, part_names_Chinese, grade, grade_pred):
    image_map, positions = process_images_conclusion_english(images_conclusion)
    WIDTH, HEIGHT = 900, 1050
    IMAGE_SIZE = (170, 160)
    FONT_PATH = "font/Arial.ttf"
    
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    
    font_title = ImageFont.truetype(FONT_PATH, 30)
    font_text = ImageFont.truetype(FONT_PATH, 20)
    
    draw.text((WIDTH // 2 - 110, 20), "Gastroscopy Report", fill="black", font=font_title)
    draw.text((50, 75), "Representative Endoscopic Images Selection:", fill="black", font=font_text)
    
    for (x, y), (img_path, part_name) in zip(positions, image_map.items()):
        img = Image.open(img_path).resize(IMAGE_SIZE)
        image.paste(img, (x, y))
        if " and " in part_name:
            part_name = part_name.replace(" and ", "/")
            draw.text((x, y + 165), part_name, fill="black", font=font_text)
        else:
            draw.text((x + 30, y + 165), part_name, fill="black", font=font_text)
    draw.text((50, 520), "Region Description:", fill="black", font=font_text)
    y_text_start = 550
    
    formatted_descriptions = []
    for description in region_descriptions:
        part_name_ch, desc = description.split('：', 1)
        translated_desc = translate_text_english(desc)
        english_part_name = part_names[part_names_Chinese.index(part_name_ch)]
        wrapped_text = wrap_text_english(f"{english_part_name}: {translated_desc}")
        formatted_descriptions.append('- ' + wrapped_text[0])
        formatted_descriptions.extend('  ' + line for line in wrapped_text[1:])
    
    for i, desc in enumerate(formatted_descriptions):
        draw.text((70, y_text_start + i * 30), desc, fill="black", font=font_text)
    
    translated_conclusion = "Diagnosis Conclusion: " + translate_text_english(conclusion.split('：')[-1])
    y_offset = y_text_start + len(formatted_descriptions) * 30 + 20
    draw.text((50, y_offset), translated_conclusion, fill="black", font=font_text)
    draw.text((50, y_offset + 40), f"Upper GI Cancer Screening: {grade.get(grade_pred, 'Unknown')}", fill="black", font=font_text)
    
    image.save("report_english.jpg")
    print("Report saved as report_english.jpg.")

def process_images_conclusion_chinese(images_conclusion):
    image_map = {img: part for part, imgs in images_conclusion.items() for img in imgs}
    num_images = len(image_map)
    
    positions = [(x, y) for y in [110, 310] for x in [50, 230, 410, 590]][:num_images]
    return image_map, positions

def wrap_text_chinese(text, max_length=33):
    return ['·' + text[i:i+max_length] if i == 0 else '  ' + text[i:i+max_length] 
            for i in range(0, len(text), max_length)]

def create_report_chinese(region_descriptions, conclusion, images_conclusion, part_names, part_names_Chinese, grade_Chinese, grade_pred):
    image_map, positions = process_images_conclusion_chinese(images_conclusion)
    WIDTH, HEIGHT = 800, 920
    IMAGE_SIZE = (170, 160)
    FONT_PATH = 'font/simsun.ttc'
    
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    
    font_title = ImageFont.truetype(FONT_PATH, 30)
    font_text = ImageFont.truetype(FONT_PATH, 20)

    draw.text((WIDTH // 2 - 110, 20), "胃镜检查报告单", fill="black", font=font_title)
    draw.text((50, 70), "内镜代表图片选择：", fill="black", font=font_text)
    
    for (x, y), (img_path, part_name) in zip(positions, image_map.items()):
        try:
            img = Image.open(img_path).resize(IMAGE_SIZE)
            image.paste(img, (x, y))
            draw.text((x + 50, y + 170), part_names_Chinese[part_names.index(part_name)], fill="black", font=font_text)
        except Exception as e:
            print(f"图片加载失败: {img_path}, 错误: {e}")
    
    draw.text((50, 520), "区域描述：", fill="black", font=font_text)
    
    y_text_start = 550
    wrapped_descriptions = sum((wrap_text_chinese(desc) for desc in region_descriptions), [])
    for i, line in enumerate(wrapped_descriptions):
        draw.text((70, y_text_start + i * 30), line, fill="black", font=font_text)
    
    y_offset = y_text_start + len(wrapped_descriptions) * 30 + 20
    draw.text((50, y_offset), conclusion, fill="black", font=font_text)
    draw.text((50, y_offset + 40), f"上消化道癌症筛查：{grade_Chinese.get(grade_pred, '未知')}", fill="black", font=font_text)
    
    image.save("report_chinese.jpg")
    print("Report saved as report_chinese.jpg.")