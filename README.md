🫁 Pneumonia Detection Using Vision Transformers
A Comparative Analysis of Transformer Architectures on Chest X-Ray Classification












🛠️ Skills Demonstrated
<p align="left"> <img src="https://img.shields.io/badge/Deep%20Learning-Vision%20Transformers-orange?style=for-the-badge" /> <img src="https://img.shields.io/badge/Models-ViT%20%7C%20Swin%20%7C%20DeiT%20%7C%20PoolFormer%20%7C%20MobileViT-blue?style=for-the-badge" /> <img src="https://img.shields.io/badge/Framework-PyTorch%20%7C%20HuggingFace-yellow?style=for-the-badge" /> <img src="https://img.shields.io/badge/Image%20Processing-Chest%20X--Ray%20%7C%20Normalization-lightblue?style=for-the-badge" /> <img src="https://img.shields.io/badge/Evaluation-F1%20%7C%20AUC%20%7C%20ROC%20%7C%20PR%20%7C%20Confusion%20Matrix-red?style=for-the-badge" /> <img src="https://img.shields.io/badge/Visualization-TSNE%20%7C%20Loss%20Curves%20%7C%20ROC%20Curves-green?style=for-the-badge" /> <img src="https://img.shields.io/badge/Training-GPU%20Accelerated-brightgreen?style=for-the-badge" /> </p>
📌 Project Overview

This project performs a comparative study of five modern Vision Transformer architectures—ViT, Swin Transformer, DeiT, PoolFormer, and MobileViT—for the task of pneumonia detection using chest X-ray images.
All models were trained on the Kaggle Pneumonia dataset, balanced via downsampling to ensure fair performance evaluation.

Using PyTorch + HuggingFace Transformers, each model was optimized with early stopping and evaluated using Accuracy, Precision, Recall, F1 Score, AUC, and rich visualizations including ROC curves, PR curves, confusion matrices, and t-SNE embeddings.

According to results, MobileViT achieved the best overall performance, offering an ideal balance between accuracy and real-time deployability.


Comparision_of_Transformers_wit…

📂 Dataset

Kaggle Chest X-Ray Pneumonia Dataset
Classes:

Normal

Pneumonia

Preprocessing included:

Class balancing

Resizing to 224×224

Normalization

Train/val/test split

🧠 Models Compared
Model	Accuracy	Precision	Recall	F1 Score	AUC
MobileViT	0.8638	0.8265	0.9897	0.9008	0.9731
ViT Base	0.8510	0.8100	0.9949	0.8930	0.9657
PoolFormer	0.8477	0.8041	1.0000	0.8914	0.9723
Swin Transformer	0.8445	0.8033	0.9948	0.8888	0.9767
DeiT	0.8333	0.7907	0.9974	0.8820	0.9703

📌 MobileViT ranks #1 overall, combining accuracy + efficiency.


Comparision_of_Transformers_wit…

🛠️ Tech Stack

PyTorch

HuggingFace Transformers

NumPy, Pandas

OpenCV

Matplotlib, Seaborn

scikit-learn (metrics)

🧪 How to Run
git clone https://github.com/yourusername/pneumonia-transformer-comparison.git
cd pneumonia-transformer-comparison
pip install -r requirements.txt
python train.py --model mobilevit


To run all models:

python train_all_models.py

📈 Visualizations Provided

✔ ROC Curves for all models
✔ Precision–Recall Curves
✔ Confusion Matrices
✔ Loss Curves
✔ t-SNE latent-space embeddings
✔ Metric comparison plots

📜 Conclusion

This study demonstrates that lightweight Vision Transformers—especially MobileViT—are highly promising for real-time pneumonia screening in clinical settings.
Their efficiency and accuracy make them suitable for deployment in low-resource healthcare environments.

🤝 Contributions

Contributions and PRs are welcome!

📄 License

MIT License
