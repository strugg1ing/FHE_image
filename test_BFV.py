import cv2
import os
import tenseal as ts
import time
import pickle
import numpy as np

# 设置 TenSEAL 上下文
context = ts.context(
    ts.SCHEME_TYPE.BFV,
    poly_modulus_degree=4096,
    plain_modulus=1032193
)
context.generate_galois_keys()
context.global_scale = 2 ** 40

def read_image_to_vector(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (64, 64))
    img_vector = img_resized.flatten() / 255.0
    return img_vector

# 图片文件夹路径
input_folder_path = "test"
output_folder_path = "test-ec"
# 如果输出文件夹不存在，则创建
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
# 加密图片向量
image_vectors = []
# 记录加密时间
start_time_0 = time.time()
# 加密图片
for filename in os.listdir(input_folder_path):
    if filename.endswith(".jpg"):
        image_path = os.path.join(input_folder_path, filename)
        img_vector = read_image_to_vector(image_path)
        enc_img = ts.bfv_vector(context, img_vector)
        image_vectors.append((filename, enc_img))  # 存储原文件名和加密向量

# 将加密后的图片保存到另一个文件夹
for i, (filename, enc_img) in enumerate(image_vectors):
    encrypted_output_path = os.path.join(output_folder_path, f"encrypted_image_{filename}.pkl")
    # 将 BFVVector 对象转换为字节流
    serialized_enc_img = enc_img.serialize()
    with open(encrypted_output_path, 'wb') as file:
        file.write(serialized_enc_img)

# 提供一张图片用于同态匹配
query_image_path = "test/ILSVRC2012_val_00004737.jpg"
query_image_vector = read_image_to_vector(query_image_path)
query_enc_image = ts.bfv_vector(context, query_image_vector)

# 记录加密时间
start_time_1 = time.time()

# 同态匹配
best_match_filename = None
best_match_mse = float('inf')  # 使用初始值表示最小的MSE

for filename, enc_img in image_vectors:
    # 计算MSE
    start_start_time = time.time()
    query_decrypted = np.array(query_enc_image.decrypt())
    enc_decrypted = np.array(enc_img.decrypt())
    mse = np.mean((query_decrypted - enc_decrypted)**2)

    # 输出MSE和单个图片匹配用时
    matching_time_single = (time.time() - start_start_time) * 1000  # 转换为毫秒
    print(f"MSE（图片{filename}）: {mse:.4f}, 图片匹配用时: {matching_time_single:.4f} 毫秒")

    if mse < best_match_mse:
        best_match_mse = mse
        best_match_filename = filename

# 计算匹配总用时
matching_time_total = (time.time() - start_time_1) * 1000  # 转换为毫秒
# 计算匹配总用时
crp_time_total = (start_time_1 - start_time_0) * 1000  # 转换为毫秒
# 输出结果
print(f"最佳匹配结果: 图片{best_match_filename}")
print(f"最佳匹配的MSE: {best_match_mse:.4f}")
print("加密总用时:", crp_time_total, "毫秒")
print("匹配总用时:", matching_time_total, "毫秒")
