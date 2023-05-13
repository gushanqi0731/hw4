import os
import cv2
import numpy as np

# 文件路径
input_dir = "/public/home/gushanqi2022/diffusers/dataset/mask1"
output_dir = "/public/home/gushanqi2022/diffusers/dataset/mask"

# alpha 通道标准化到 0 到 1 之间，将 RGB 通道标准化到 -1 到 1 之间
alpha_max = 255.0
img_max = 127.5
img_min = -127.5

# 循环处理每张图片
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        # 读取图片
        mask_path = os.path.join(input_dir, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / alpha_max
        img_path = mask_path.replace("mask1", "img").replace(".png", ".jpg")
        img = cv2.imread(img_path).astype(np.float32)
        
        # 将处理过的 RGB 通道与 mask 相乘
        mask_alpha = np.expand_dims(mask[:, :, 3], axis=2)
        img_rgb = (img - img_min) / (img_max - img_min)
        img_rgb *= np.expand_dims(mask_alpha, axis=2)
        
        # 将 RGB 通道和掩码连接在一起，这就是控制网络的输入
        mask[:, :, :3] *= np.expand_dims(mask_alpha, axis=2)
        input_data = np.concatenate((img_rgb, mask[:, :, :3]), axis=2)
        
        # 将可视化的图像反转归一化为输入时的原始范围
        output_data = ((input_data + 1) * img_max).astype(np.uint8)
        
        # 保存图片
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, output_data)
