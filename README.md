# Controllable Category Diversity Framework (CCDF)
Code Repo for Paper: "On Practical Diversified Recommendation with Controllable Category Diversity Framework"

# Prerequisites
The following setup is tested and it is working:
- Python == 3.6.0
- Tensorflow == 1.15.0
- Tensorboard == 2.10.1
- Cuda >= 10.2

# Usage
Scripts to train or test DeepU2C model on alibaba/taobao dataset:
```text
python src/main.py \
        -p train|test \
        --batch_size 100 \
        --dataset taobao|alibaba \
        --lambda 1.0 \
        --m 0.4 \
        --embedding_dim 32 \
        --hidden_size 32 \
        --cuda
```


# Abstract 
Recommender systems have made significant strides in various industries, primarily driven by extensive efforts to enhance recommendation accuracy. However, this pursuit of accuracy has inadvertently given rise to echo chamber/filter bubble effects. Especially in industry, it could impair user's experiences and prevent user from accessing a wider range of items.
One of the solutions is to take diversity into account. However, most of existing works focus on user's explicit preferences, while rarely explore user's non-interaction preferences. These neglected non-interaction preferences are especially important for broadening user's interests in alleviating echo chamber/filter bubble effects.
Therefore, in this paper, we first define diversity as two distinct definitions, i.e., user-explicit diversity (U-diversity) and user-item non-interaction diversity (N-diversity) based on user historical behaviors. Then, we propose a succinct and effective method, named as Controllable Category Diversity Framework (CCDF) to achieve both high U-diversity and N-diversity simultaneously.
Specifically, CCDF consists of two stages, User-Category Matching and Constrained Item Matching. The User-Category Matching utilizes the DeepU2C model and combined loss to predict user's preferences in categories. These categories will be used as trigger information in Constrained Item Matching.
Offline experimental results show that our proposed DeepU2C outperforms state-of-the-art diversity-oriented methods, especially on N-diversity task. The whole framework is validated in a real-world production environment by conducting online A/B testing. The improved conversion rate and diversity metrics demonstrate the superiority of our proposed framework in industrial applications.
Further analysis supports the complementary effects between recommendation and search that diversified recommendation is able to effectively help users to discover new needs, and then inspire them to refine their demands in search.


# Citation
If you use the code in this repository in your paper, please consider citing:
```
@inproceedings{zhang2024ccdf,
  title={On Practical Diversified Recommendation with Controllable Category Diversity Framework},
  author={Tao Zhang, Luwei Yang and Zhibo Xiao, Wen Jiang and Wei Ning}
}
```