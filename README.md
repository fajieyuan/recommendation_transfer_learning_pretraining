# recommendation_transfer_learning_pre-training:  https://github.com/westlake-repl/Recommendation-Systems-without-Explicit-ID-Features-A-Literature-Review


1 Parameter-Efficient Transfer from Sequential Behaviors for User Modeling and Recommendation SIGIR2020 https://arxiv.org/abs/2001.04253

Keywords: self-supervise learning, user sequential behaviors, pretraining, transfer learning, user representation, user profile prediction, cold-start problem

(1) We are the first to evidence that self-supervised user behavior pre-training helps many downstream tasks. 

(2) We are also the first to provide user profile prediction as a way to show the universal property of user representation

(3) We release a large-scale public dataset for user representation transer learning and source code.


Inductive transfer learning has had a big impact on computer vision and NLP domains but has not been used in the area of recommender systems. Even though there has been a large body of research on generating recommendations based on modeling user-item interaction sequences, few of them attempt to represent and transfer these models for serving downstream tasks where only limited data exists.  In this paper, we delve on the task of effectively learning a single user representation that can be applied to a diversity of tasks, from cross-domain recommendations to user profile predictions. Fine-tuning a large pre-trained network and adapting it to downstream tasks is an effective way to solve such tasks. However, fine-tuning is parameter inefficient considering that an entire model needs to be re-trained for every new task. To overcome this issue, we develop a parameter efficient transfer learning architecture, termed as PeterRec, which can be configured on-the-fly to various downstream tasks. Specifically, PeterRec allows the pre-trained parameters to remain unaltered during fine-tuning by injecting a series of re-learned neural networks, which are small but as expressive as learning the entire network. We perform extensive experimental ablation to show the effectiveness of the learned user representation in five downstream tasks. Moreover, we show that PeterRec performs efficient transfer learning in multiple domains, where it achieves comparable or sometimes better performance relative to fine-tuning the entire model parameters. Codes and datasets are available at this https URL


2 One Person, One Model, One World: Learning Continual User Representation without Forgetting SIGIR2021 https://arxiv.org/abs/2009.13724

Keywords: self-supervise learning, lifelong learning, pretraining, transfer learning, finetuning, user representation, user profile prediction, cold-start problem

(1) We are the first to propose universal lifelong user representation learning mechanism for recommender system

(2) We are the first to clearly demonstrate the catastrophic forgetting and over-parameterization issues in recommender sytem.

(3) We release the dataset for lifelong user representation learning and source code.


Learning user representations is a vital technique toward effective user modeling and personalized recommender systems. Existing approaches often derive an individual set of model parameters for each task by training on separate data. However, the representation of the same user potentially has some commonalities, such as preference and personality, even in different tasks. As such, these separately trained representations could be suboptimal in performance as well as inefficient in terms of parameter sharing. 
In this paper, we delve on research to continually learn user representations task by task, whereby new tasks are learned while using partial parameters from old ones. A new problem arises since when new tasks are trained, previously learned parameters are very likely to be modified, and as a result, an artificial neural network (ANN)-based model may lose its capacity to serve for well-trained previous tasks forever, this issue is termed catastrophic forgetting. To address this issue, we present \emph{Conure} the first \underline{con}tinual, or lifelong, \underline{u}ser \underline{re}presentation learner -- i.e., learning new tasks over time without forgetting old ones. Specifically, we propose iteratively removing less important weights of old tasks in a deep user representation model, motivated by the fact that neural network models are usually over-parameterized. In this way, we could learn many tasks with a single model by reusing the important weights, and modifying the less important weights to adapt to new tasks. We conduct extensive experiments on two real-world datasets with nine tasks and show that \emph{Conure} largely exceeds the standard model that does not purposely preserve such old "knowledge", and performs competitively or sometimes better than models which are trained either individually for each task or simultaneously by merging all task data.


3 Learning Transferable User Representations with Sequential Behaviors via Contrastive Pre-training ICDM2021 https://fajieyuan.github.io/papers/ICDM2021.pdf

Keywords: contrative learnng, self-supervise learning, transfer learning,  pretraining, finetuning, user representation, user profile prediction, cold-start problem

Learning effective user representations from sequen- tial user-item interactions is a fundamental problem for recom- mender systems (RS). Recently, several unsupervised methods focusing on user representations pre-training have been explored. In general, these methods apply similar learning paradigms by first corrupting the behavior sequence, and then restoring the original input with some item-level prediction loss functions. Despite its effectiveness, we argue that there exist important gaps between such item-level optimization objective and user-level representations, and as a result, the learned user representations may only lead to sub-optimal generalization performance. In this paper, we propose a novel self-supervised pre-training frame- work, called CLUE, which stands for employing Contrastive Learning for modeling sequence-level User rEpresentation. The core idea of CLUE is to regard each user behavior sequence as a whole and then construct the self-supervision signals by trans- forming the original user behaviors by data augmentations (DA). Specifically, we employ two Siamese (weight-sharing) networks to learn the user-oriented representations, where the optimization goal is to maximize the similarity of learned representations of the same user by these two encoders. More importantly, we perform careful investigation of the impacts of view generating strategies for user behavior inputs from a more comprehensive perspective, including processing sequential behaviors by explicit DA strategies and employing dropout as implicit DA. To verify the effectiveness of CLUE, we perform extensive experiments on several user-related tasks with different scales and characteristics. Our experimental results show that the user representations learned by CLUE surpass existing item-level baselines under several evaluation protocols.

4 User-specific Adaptive Fine-tuning for Cross-domain Recommendations TKDE2021 https://arxiv.org/pdf/2106.07864.pdf

Keywords: adaptive finetuning, pretraining, cold-start problem, cross-domain recommendation

Making accurate recommendations for cold-start users has been a longstanding and critical challenge for recommender systems (RS). Cross-domain recommendations (CDR) offer a solution to tackle such a cold-start problem when there is no sufficient data for the users who have rarely used the system. An effective approach in CDR is to leverage the knowledge (e.g., user representations) learned from a related but different domain and transfer it to the target domain. Fine-tuning works as an effective transfer learning technique for this objective, which adapts the parameters of a pre-trained model from the source domain to the target domain. However, current methods are mainly based on the global fine-tuning strategy: the decision of which layers of the pre-trained model to freeze or fine-tune is taken for all users in the target domain. In this paper, we argue that users in RS are personalized and should have their own fine-tuning policies for better preference transfer learning. As such, we propose a novel User-specific Adaptive Fine-tuning method (UAF), selecting which layers of the pre-trained network to fine-tune, on a per-user basis. Specifically, we devise a policy network with three alternative strategies to automatically decide which layers to be fine-tuned and which layers to have their parameters frozen for each user. Extensive experiments show that the proposed UAF exhibits significantly better and more robust performance for user cold-start recommendation.

## 5 TransRec: Learning Transferable Recommendation from Mixture-of-Modality Feedback     Arxiv: https://arxiv.org/pdf/2206.06190.pdf 

Keywords: tranfer learning, pre-training, mixture-of-modality, content-based recommendation, general-purpose recommender system

Learning big models and then transfer has become the de facto practice in computer vision (CV) and natural language processing (NLP). However, such unified
paradigm is uncommon for recommender systems (RS). A critical issue that hampers this is that standard recommendation models are built on unshareable identity
data, where both users and their interacted items are represented by unique IDs. In this paper, we study a novel scenario where user’s interaction feedback involves
mixture-of-modality (MoM) items. We present TransRec, a straightforward modification done on the popular ID-based RS framework. TransRec directly learns
from MoM feedback in an end-to-end manner, and thus enables effective transfer  learning under various scenarios without relying on overlapped users or items.
We empirically study the transferring ability of TransRec across four different real-world recommendation settings. Besides, we study its effects by scaling the size of source and target data. Our results suggest that learning recommenders from MoM feedback provides a promising way to realize universal recommender systems. Our code and datasets will be made available.


We have also released large-scale dataset (over 1 million user clicking behaviors) for performing transfer learning of user preference in recommendation field
https://github.com/fajieyuan/recommendation_dataset_pretraining

