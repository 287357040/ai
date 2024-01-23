from flask import Flask, request, render_template
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from flask import jsonify
import torch
import io
import base64

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Step 1: Choose a pre-trained model architecture
model_name = "Jim320/clothing-classification"

# Step 3: Define a imageprocessor
imageProcessor = AutoImageProcessor.from_pretrained(model_name)

# Step 4: Load the pre-trained model
try:
    model = AutoModelForImageClassification.from_pretrained(model_name)
except OSError:
    print(f"Model not found locally. Downloading {model_name}...")
    model = AutoModelForImageClassification.from_pretrained(model_name)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file part")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction="No selected file")

    if file:
        contents = file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 预处理图像并生成输入
        inputs = imageProcessor(images=image, return_tensors="pt", padding="max_length", truncation=True)

        # 进行推理
        outputs = model(**inputs)

        # 获取所有标签及其对应的概率分数，构建字典
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
        labels = [i for i in range(len(probabilities))]  # 假设标签是 0 到 n-1 的整数，可以替换成你的实际标签
        label_name_map = {
            "0": "fullback",
            "1": "fullfront",
            "2": "lower",
            "3": "nohead",
            "4": "noperson",
            "5": "upper"
        }

        # 将标签映射到名称
        label_names = [label_name_map[str(label)] for label in labels]

        # 构建包含标签名称和概率的列表
        label_score_list = [{'label': label_name, 'score': score} for label_name, score in zip(label_names, probabilities)]

        # 按照分数进行排序
        sorted_label_score_list = sorted(label_score_list, key=lambda x: x['score'], reverse=True)


        # 将输出信息传递给模板
        return jsonify({'sorted_label_score_list': sorted_label_score_list})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)