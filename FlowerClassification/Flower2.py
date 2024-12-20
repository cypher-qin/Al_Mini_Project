import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split
from transformers import ViTForImageClassification, Trainer, TrainingArguments, ViTImageProcessor

data = pd.read_csv('F:/机器学习大作业/2024秋6系ML-本科-团队1-花朵分类/train/train.csv')
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=17)
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
# 建立类别图谱
flower_types = data['label'].unique()
label_map = {label: idx for idx, label in enumerate(flower_types)}
ati_label_map = {idx: label for idx, label in enumerate(flower_types)}

# transform = transforms.Compose([
#    transforms.RandomHorizontalFlip(),
#    transforms.RandomRotation(30),
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])


class FlowerDataset(Dataset):
    def __init__(self, img_dir, train=True, feature_extractor=None):
        self.feature_extractor = feature_extractor
        self.img_dir = img_dir
        if train:
            self.data = train_data
        else:
            self.data = test_data
        self.label_map = label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, f"{self.data.iloc[idx, 0]}.jpg")
        image_PIL = Image.open(img_name).convert("RGB")
        label = self.data.iloc[idx, 1]
        label = self.label_map[label]
        image_dict = self.feature_extractor(images=image_PIL, return_tensors="pt")
        return {
            'pixel_values': image_dict['pixel_values'].squeeze(0),
            'label': label
        }


# 调用GPU
if torch.cuda.is_available():
    print("GPU working!")
    device = torch.device("cuda")
    model.to(device)
# 创建训练集和测试集的 Dataset 对象
train_dataset = FlowerDataset(img_dir='F:/机器学习大作业/2024秋6系ML-本科-团队1-花朵分类/train/images/images',
                              feature_extractor=feature_extractor)

test_dataset = FlowerDataset(img_dir='F:/机器学习大作业/2024秋6系ML-本科-团队1-花朵分类/train/images/images',
                             feature_extractor=feature_extractor, train=False)

training_args = TrainingArguments(
    output_dir='./results',  # 输出文件夹
    num_train_epochs=5,
    per_device_train_batch_size=8,  # 每个设备的批次大小
    per_device_eval_batch_size=8,  # 每个设备的评估批次大小
    warmup_steps=500,  # 学习率预热步数
    weight_decay=0.01,  # 权重衰减
    logging_dir=None,  # 不生成日志文件
    logging_steps=-1,  # 不输出日志
    disable_tqdm=True,  # 关闭进度条
    evaluation_strategy="epoch",  # 每个 epoch 评估一次
    save_strategy="epoch",  # 每个 epoch 保存一次
    load_best_model_at_end=True,  # 在训练结束时加载最佳模型
)


def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=1)  # 选择概率最大的类别
    accuracy = accuracy_score(labels, preds)
    return {'accuracy': accuracy}


# 4. 创建 Trainer
trainer = Trainer(
    model=model,  # 训练的模型
    args=training_args,  # 训练的参数
    train_dataset=train_dataset,  # 训练数据集
    eval_dataset=test_dataset,  # 测试数据集
    compute_metrics=compute_metrics
)

# 5. 训练模型
trainer.train()
model.save_pretrained('./saved_model')


def predict_image(image_path, model, feature_extractor):
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class


image_folder = 'F:/机器学习大作业/2024秋6系ML-本科-团队1-花朵分类/test/images/images'
predictions = []

# 遍历图片文件夹中的所有图片进行预测
for img_name in sorted(os.listdir(image_folder), key=lambda x: int(os.path.splitext(x)[0])):
    img_path = os.path.join(image_folder, img_name)
    img_id = os.path.splitext(img_name)[0]  # 获取文件名的数字部分
    label_index = predict_image(img_path, model, feature_extractor)  # 假设predict_image是定义好的预测函数
    label_string = ati_label_map[label_index]  # 将标签索引转换为标签字符串
    predictions.append({"id": img_id, "label": label_string})


# 将结果保存到CSV文件
df = pd.DataFrame(predictions)
df.to_csv("submission.csv", index=False)
