# SR-FFRT: STRUCTURAL REPARAMETERIZATION FOR FAST FACIAL EXPRESSION RECOGNITION
## 项目简介
SR-FFRT 是一个专注于实时面部表情识别的创新深度学习模型。该模型采用结构重参数化（Structural Reparameterization）和边界去除模块（Boundary Removal Block），以提高表情表示的精度和效率。
## 论文摘要
我们提出了 SR-FFRT 模型，通过结构重参数化减少计算复杂度，保持训练时的特征提取能力。此外，我们还引入了双损失函数，以解决面部表情数据集中类不平衡问题。实验结果表明，SR-FFRT 模型在多个基准测试中达到了最先进的性能，并显著提高了推理速度。
## 模型架构
- **结构重参数化**: 提高了推理效率，保持了模型的高效特征提取。
- **边界去除模块（Boundary Removal Block）**: 通过减轻特征图边缘的侵蚀问题，提高面部表情表示的准确性。
- **Inception-FFN**: 基于 Inception 风格的深度学习模块，集成了 Vision Transformers 的优势，提升了面部表情识别的准确性。
## 方法
SR-FFRT 模型由四个主要阶段组成，每个阶段都包含注意力混合器和 Inception-FFN 模块。该架构通过层次化的方法，逐层提取不同尺度的特征，以增强面部表情识别的准确性。
## 实验结果
实验在 RAF-DB、AffectNet 和 FERPlus 数据集上进行，SR-FFRT 模型在这些数据集上的表现优于当前最先进的表情识别方法，并在推理速度方面显示出明显的优势。
## 参考文献
- Ce Zheng, Matias Mendieta, and Chen Chen, “Poster: A pyramid cross-fusion transformer network for facial expression recognition,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.
- Jiawei Mao, Rui Xu, Xuesong Yin, Yuanqi Chang, Binling Nie, and Aibin Huang, “Poster v2: A simpler and stronger facial expression recognition network,” arXiv preprint arXiv:2301.12149, 2023.
- Azmine Toushik Wasi, Karlo Šerbetar, Raima Islam, Taki Hasan Rafi, and Dong-Kyu Chae, “Arbex: Attentive feature extraction with reliability balancing for robust facial expression learning,” arXiv preprint arXiv:2305.01486, 2023.
