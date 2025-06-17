# Tối ưu phân bổ tài nguyên và Gia tăng độ tin cậy trong mạng tích hợp Sub6GHz/mmWave bằng Học sâu tăng cường
**English below**

## Mục lục
- [Giới thiệu](##Giới-thiệu)
- [Tính năng](#tính-năng)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Tập dữ liệu](#tập-dữ-liệu)
- [Huấn luyện](#huấn-luyện)
- [Đánh giá](#đánh-giá)
- [Kết quả](#kết-quả)

---

## Giới thiệu

Dự án này tập trung vào việc tối ưu phân bổ tài nguyên và tăng cường độ tin cậy trong mạng không dây tích hợp Sub6GHz/mmWave bằng phương pháp Học tăng cường sâu (DRL). Chúng tôi đề xuất một phương pháp mới mang tên SACPA (Soft Actor Critic with Power Allocation), giúp đồng thời học được giao diện truyền tối ưu (Sub6GHz hoặc mmWave) và công suất truyền cho từng thiết bị trong mạng.

---

## Tính năng

- Mô phỏng mạng tích hợp Sub6GHz/mmWave theo API Gymnasium và StableBaselines3
- Phân bổ tài nguyên dựa trên học tăng cường sâu
- Cơ chế tăng cường độ tin cậy

---

## Cài đặt

- Tải mã nguồn:
```bash
git clone https://github.com/ductaingn/Power-Allocation
```
- Cài đặt thư viện:
```bash
cd Power-Allocation
pip install -r requirements.txt
```

## Hướng dẫn sử dụng
### Tập dữ liệu
- Sinh tập dữ liệu:
  - Sinh vị trí thiết bị (hỗ trợ tối đa 15 thiết bị với vị trí định sẵn): Sử dụng `generate_devices_positions()` trong `Power-Allocation/environment/Environment.py`.
  - Sinh $\tilde{h}_{kn}^v$: Sử dụng `generate_h_tilde()` trong `Power-Allocation/environment/Environment.py`.

### Huấn luyện
- Huấn luyện với một thuật toán:
  1. Chỉnh sửa cấu hình trong `train_config.yaml`
  2. Chạy:
```bash
cd Power-Allocation
python train.py
``` 
- Huấn luyện nhiều thuật toán (benchmark):
  1. Chỉnh sửa cấu hình trong `train_config.yaml`
  2. Chạy:
```bash
cd Power-Allocation
python benchmark.py
```

### Đánh giá
- Sử dụng các hàm trong `Power-Allocation/helper.py` để trích xuất và xử lý dữ liệu từ Wandb.

## Kết quả 
Các kết quả thí nghiệm cho thấy SACPA vượt trội hơn rõ rệt so với các thuật toán cơ sở về phần thưởng và độ tin cậy. Đặc biệt, SACPA đạt tỷ lệ thành công trung bình cao hơn trong nhiều điều kiện mạng khác nhau.


# Resource Allocation Optimization and Reliability Enhancement in Sub6GHz/mmWave Integrated Network with Deep Reinforcement Learning

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)

## Introduction

This project focuses on optimizing resource allocation and improving reliability in an integrated Sub6GHz/mmWave wireless network using Deep Reinforcement Learning. We propose a novel approach called SACPA (Soft Actor Critic with Power Allocation), which simultaneously learns both the optimal interface (Sub6GHz or mmWave) and transmission power for each device in the network.


## Features

- Sub6GHz/mmWave integrated network simulation with Gymnasium and StableBaselines3 API
- Deep reinforcement learning-based resource allocation
- Reliability enhancement mechanisms

## Installation
- Clone this repository:
```
git clone https://github.com/ductaingn/Power-Allocation
```
- Install dependencies:

``` bash
cd Power-Allocation
pip install -r requirements.txt 
```

## Usage

### Dataset

- Generate dataset
  - Generate device positions (currently only supports upto 15 devices with predefined position): Use `generate_devices_positions()` function in `Power-Allocation/environment/Environment.py`
  - Generate $\tilde{h}_{kn}^v$: Use `generate_h_tilde()` function in `Power-Allocation/environment/Environment.py`

### Training

- Training with one algorithm:
  1. Change algorithm configurations in `Power-Allocation\train_config.yaml`
  2. Run 
```bash
cd Power-Allocation 
python train.py
```

- Training multiple algorithm:
  1. Change algorithm configurations in `Power-Allocation\train_config.yaml`
  2. Run
```bash
cd Power-Allocation
python benchmark.py
```

### Evaluation
- Use helper functions in `Power-Allocation/helper.py` to get the data from Wandb logs and process data with functions in `Power-Allocation/helper.py` to get the results.

## Results

Our experiments demonstrate that SACPA (LearnInterfaceAndPower) significantly outperforms baseline algorithms in both reward and reliability metrics. Particularly, SACPA achieves better average success rates under varying network conditions.

## Contact
For any questions or contributions, please contact:
Duc Tai Nguyen - ductaingn.015203@gmail.com