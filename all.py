import os
import cv2
import torch
import base64
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------
# 初始化YOLO模型
# ------------------------
yolo_model = YOLO("best.pt")  # 替换为你的YOLO模型路径

# ------------------------
# 加载你训练好的Qwen模型
# ------------------------
tokenizer = AutoTokenizer.from_pretrained(
    "./output/Qwen3-1.7B/checkpoint-8800", 
    use_fast=False, 
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "./output/Qwen3-1.7B/checkpoint-8800", 
    device_map="auto", 
    torch_dtype=torch.bfloat16
)

# ------------------------
# YOLO检测函数
# ------------------------
def yolo_detect(image_path):
    results = yolo_model(image_path)
    detected_diseases = []
    boxes_info = []  # 存储每个框的信息 (label, area)

    # 英文到中文映射
    en2zh = {
        "curl_stage1": "曲叶病初期",
        "curl_stage2": "曲叶病中期",
        "healthy": "健康",
        "leaf_enation": "根结线虫病",
        "sooty": "白霉病"
    }

    # 中文疾病对应固定颜色（BGR）
    disease_color_map = {
        "曲叶病初期": (35, 186, 197),
        "曲叶病中期": (35, 178, 197),
        "健康": (238, 193, 134),
        "根结线虫病": (165, 118, 28),
        "白霉病": (255, 97, 0)
    }
    
    for result in results:
        boxes = result.boxes
        for box, cls in zip(boxes.xyxy, boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            label_en = result.names[int(cls)]
            label = en2zh.get(label_en, label_en)  # 转中文
            detected_diseases.append(label)
            
            # 框面积
            area = (x2 - x1) * (y2 - y1)
            boxes_info.append({"label": label, "area": area})
            
            # 固定颜色
            color = disease_color_map.get(label, (0, 0, 0))  # 默认黑色
            
            # 画框（去掉文字）
            cv2.rectangle(result.orig_img, (x1, y1), (x2, y2), color, 2)
    
    # 图片转base64
    _, buffer = cv2.imencode('.jpg', result.orig_img)
    img_base64 = base64.b64encode(buffer).decode()
    return detected_diseases, img_base64, boxes_info


# ------------------------
# Qwen生成防治方法
# ------------------------
# ------------------------
# Qwen生成防治方法（修改版）
# ------------------------
def qwen_generate(disease_name):
    if disease_name == "健康":
        return "叶片该位置健康。"  # 固定文本
    prompt = f"{disease_name}的防治措施如下："
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


# ------------------------
# 生成HTML文件（修改版）
# ------------------------
def generate_html(original_image_path, output_html="result.html"):
    # YOLO检测结果
    diseases, yolo_img_base64, boxes_info = yolo_detect(original_image_path)
    
    # 原图转base64
    orig_img = cv2.imread(original_image_path)
    _, buffer = cv2.imencode('.jpg', orig_img)
    orig_img_base64 = base64.b64encode(buffer).decode()
    
    # 去重保持顺序
    unique_diseases = []
    seen = set()
    for d in diseases:
        if d not in seen:
            unique_diseases.append(d)
            seen.add(d)
    
    # 疾病对应字体颜色 HEX
    disease_color_map = {
        "曲叶病初期": "#C5BA23",
        "曲叶病中期": "#C5B223",
        "健康": "#86C1EE",
        "根结线虫病": "#1C76A5",
        "白霉病": "#0061FF"
    }

    disease_cards = []
    for disease in unique_diseases:
        text_color = disease_color_map.get(disease, "#333333")
        prevention_text = qwen_generate(disease)
        disease_cards.append(f"""
        <div class="card" style="background-color:#FFFFFF;">
            <h3 style="color:{text_color};">{disease}</h3>
            <p>{prevention_text}</p>
        </div>
        """)

    # 根据框面积判断最严重疾病（排除健康）
    non_healthy_boxes = [b for b in boxes_info if b["label"] != "健康"]
    if non_healthy_boxes:
        most_severe_disease = max(non_healthy_boxes, key=lambda x: x["area"])["label"]
        severity_note = f"""
        <div class="severity-note">
            ⚠️ 该叶片中 {most_severe_disease} 更为严重，建议以该疾病的防治方法为主。
        </div>
        """
    else:
        severity_note = ""

    html_content = f"""
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>棉叶病害智能检测</title>
        <style>
            body {{
                font-family: 'Comic Sans MS', Arial, sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(to bottom right, #DFF7E4, #FFF7D9);
                color: #333;
                text-align: center;
            }}
            .container {{
                width: 90%;
                margin: 20px auto;
            }}
            h1 {{
                font-size: 40px;
                margin-bottom: 40px;
                color: #3E8E41;
                text-shadow: 2px 2px 5px rgba(0,0,0,0.2);
                border-bottom: 4px dashed #FF9F80;
                display: inline-block;
                padding-bottom: 10px;
            }}
            .images {{
                display: flex;
                gap: 20px;
                justify-content: center;
                margin-bottom: 40px;
                flex-wrap: wrap;
            }}
            .images img {{
                max-width: 15%;
                border-radius: 15px;
                border: 4px solid #FFECB3;
                box-shadow: 0 6px 15px rgba(0,0,0,0.15);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }}
            .images img:hover {{
                transform: scale(1.1);
                box-shadow: 0 10px 25px rgba(0,0,0,0.25);
            }}
            .results h2 {{
                margin-bottom: 20px;
                font-size: 30px;
                color: #3E8E41;
            }}
            .card {{
                padding: 15px 20px;
                margin-bottom: 15px;
                border-radius: 15px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.15);
                transition: transform 0.3s, box-shadow 0.3s;
            }}
            .card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 20px rgba(0,0,0,0.25);
            }}
            .card h3 {{
                margin: 0 0 10px 0;
            }}
            .card p {{
                margin: 0;
                color: #555;
            }}
            /* 红色呼吸警告框 */
            .severity-note {{
                margin-top: 20px;
                padding: 15px;
                border-radius: 12px;
                background: rgba(255, 82, 82, 0.15);
                border: 2px solid #FF5252;
                color: #D32F2F;
                font-size: 18px;
                font-weight: bold;
                animation: pulse 2s infinite;
            }}
            @keyframes pulse {{
                0% {{ box-shadow: 0 0 5px rgba(255,82,82,0.6); }}
                50% {{ box-shadow: 0 0 20px rgba(255,82,82,1); }}
                100% {{ box-shadow: 0 0 5px rgba(255,82,82,0.6); }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>棉叶病害智能检测 🌿</h1>
            <div class="images">
                <img src="data:image/jpeg;base64,{orig_img_base64}" alt="原图">
                <img src="data:image/jpeg;base64,{yolo_img_base64}" alt="YOLO检测结果">
            </div>
            <div class="results">
                <h2>检测结果及防治方法</h2>
                {"".join(disease_cards)}
                {severity_note}
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"高亮警告版HTML文件已生成：{output_html}")







# ------------------------
# 主函数
# ------------------------
if __name__ == "__main__":
    generate_html("test.jpg")  # 替换为你的图片路径