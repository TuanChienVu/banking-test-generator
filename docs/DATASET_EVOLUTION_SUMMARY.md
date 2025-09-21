# 📊 Dataset Evolution Summary (V4 Baseline)

## Overview
Hệ thống dataset đã tiến hoá qua nhiều phiên bản và hiện tại baseline là V4 Natural Diversity. V4 tập trung vào tính đa dạng tự nhiên, giữ 100% uniqueness, 25% security coverage và cấu trúc tối ưu cho huấn luyện mô hình.

## Current Baseline — V4 Natural Diversity
- Splits: train / val / test
- Uniqueness: 100%
- Security Coverage: 25%
- Compliance: ISO-27001, GDPR, PCI-DSS, MASVS
- Phân tích và báo cáo: dùng các script V4 trong thư mục scripts

### Vị trí dữ liệu
```
datasets/final/testcase_generation/diverse_v4_natural_diversity_v1/
├── train.json
├── val.json
└── test.json
```

### Phân tích và báo cáo
```bash
# Phân tích chi tiết V4
python scripts/analyze_v4_detailed.py

# Tạo report hợp nhất và biểu đồ
python scripts/generate_analysis_reports.py
python scripts/generate_visualization_charts.py
```

## Legacy (Lịch sử)
Các phiên bản V1/V2/V3 trước đây đã được lưu trữ (archived). Tài liệu so sánh chi tiết giữa các phiên bản cũ đã được loại bỏ để tránh gây nhầm lẫn; vui lòng sử dụng V4 cho mọi phân tích/báo cáo hiện tại.

- V1: format tốt, chưa có security
- V2: tăng security, nhưng đã lỗi thời so với baseline hiện tại
- V3: generator sản xuất dataset đa dạng; hiện vẫn dùng để “generate”, nhưng phân tích/báo cáo theo baseline V4

## Gợi ý quy trình (hiện tại)
1) Generate (V3)
```bash
cd src/generators
python generate_diverse_dataset_v3.py
```
2) Analyze/Report (V4)
```bash
python scripts/analyze_v4_detailed.py
python scripts/generate_analysis_reports.py
python scripts/generate_visualization_charts.py
```

## Lưu ý
- Tránh dùng tài liệu/so sánh cũ (V2/V3) trong báo cáo chính. Tất cả kết quả nên dựa trên dataset V4 và các báo cáo V4.
- Nếu cần tham chiếu lịch sử, xem các mục “archive” tương ứng trong repo.
