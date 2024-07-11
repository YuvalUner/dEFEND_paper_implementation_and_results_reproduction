This repository contains an implementation for the deep learning model described in ["dEFEND: Explainable Fake News Detection"](https://dl.acm.org/doi/10.1145/3292500.3330935) 
by Kai Shu et al. (2019).\
\
The model is implemented in PyTorch and was trained (in the original paper) on the [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) dataset.\
Due to issues accessing the comments of the dataset, we provide here a modified version with comments generated using AI.\
\
Please refer to the provided Jupyter notebook for a demonstration of the model.\
Please refer to the original paper for more details on the model and the dataset.

# References
```
@inproceedings{10.1145/3292500.3330935,
author = {Shu, Kai and Cui, Limeng and Wang, Suhang and Lee, Dongwon and Liu, Huan},
title = {dEFEND: Explainable Fake News Detection},
year = {2019},
isbn = {9781450362016},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3292500.3330935},
doi = {10.1145/3292500.3330935},
abstract = {In recent years, to mitigate the problem of fake news, computational detection of fake news has been studied, producing some promising early results. While important, however, we argue that a critical missing piece of the study be the explainability of such detection, i.e., why a particular piece of news is detected as fake. In this paper, therefore, we study the explainable detection of fake news. We develop a sentence-comment co-attention sub-network to exploit both news contents and user comments to jointly capture explainable top-k check-worthy sentences and user comments for fake news detection. We conduct extensive experiments on real-world datasets and demonstrate that the proposed method not only significantly outperforms 7 state-of-the-art fake news detection methods by at least 5.33\% in F1-score, but also (concurrently) identifies top-k user comments that explain why a news piece is fake, better than baselines by 28.2\% in NDCG and 30.7\% in Precision.},
booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
pages = {395–405},
numpages = {11},
keywords = {explainable machine learning, fake news, social network},
location = {Anchorage, AK, USA},
series = {KDD '19}
}
```
```
@article{shu2018fakenewsnet,
  title={FakeNewsNet: A Data Repository with News Content, Social Context and Dynamic Information for Studying Fake News on Social Media},
  author={Shu, Kai and  Mahudeswaran, Deepak and Wang, Suhang and Lee, Dongwon and Liu, Huan},
  journal={arXiv preprint arXiv:1809.01286},
  year={2018}
}
```
```
@article{shu2017fake,
  title={Fake News Detection on Social Media: A Data Mining Perspective},
  author={Shu, Kai and Sliva, Amy and Wang, Suhang and Tang, Jiliang and Liu, Huan},
  journal={ACM SIGKDD Explorations Newsletter},
  volume={19},
  number={1},
  pages={22--36},
  year={2017},
  publisher={ACM}
}
```
```
@article{shu2017exploiting,
  title={Exploiting Tri-Relationship for Fake News Detection},
  author={Shu, Kai and Wang, Suhang and Liu, Huan},
  journal={arXiv preprint arXiv:1712.07709},
  year={2017}
}
```
