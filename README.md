# recommendation_transfer_learning_pretraining
Pre-training and Transfer learning papers and datasets for recommendation 

Several of our research papers that apply pre-training, transfer learning and representation learning for cross-domain recommendation and user profile prediction:

1 Parameter-Efficient Transfer from Sequential Behaviors for User Modeling and Recommendation SIGIR2020 https://arxiv.org/abs/2001.04253

Keywords: self-supervise learning, user sequential behaviors, pretraining, transfer learning, user representation, user profile prediction, cold-start problem

Inductive transfer learning has had a big impact on computer vision and NLP domains but has not been used in the area of recommender systems. Even though there has been a large body of research on generating recommendations based on modeling user-item interaction sequences, few of them attempt to represent and transfer these models for serving downstream tasks where only limited data exists.  In this paper, we delve on the task of effectively learning a single user representation that can be applied to a diversity of tasks, from cross-domain recommendations to user profile predictions. Fine-tuning a large pre-trained network and adapting it to downstream tasks is an effective way to solve such tasks. However, fine-tuning is parameter inefficient considering that an entire model needs to be re-trained for every new task. To overcome this issue, we develop a parameter efficient transfer learning architecture, termed as PeterRec, which can be configured on-the-fly to various downstream tasks. Specifically, PeterRec allows the pre-trained parameters to remain unaltered during fine-tuning by injecting a series of re-learned neural networks, which are small but as expressive as learning the entire network. We perform extensive experimental ablation to show the effectiveness of the learned user representation in five downstream tasks. Moreover, we show that PeterRec performs efficient transfer learning in multiple domains, where it achieves comparable or sometimes better performance relative to fine-tuning the entire model parameters. Codes and datasets are available at this https URL


2 One Person, One Model, One World: Learning Continual User Representation without Forgetting SIGIR2021 https://arxiv.org/abs/2009.13724

Keywords: self-supervise learning, lifelong learning, pretraining, transfer learning, finetuning, user representation, user profile prediction, cold-start problem

Learning user representations is a vital technique toward effective user modeling and personalized recommender systems. Existing approaches often derive an individual set of model parameters for each task by training on separate data. However, the representation of the same user potentially has some commonalities, such as preference and personality, even in different tasks. As such, these separately trained representations could be suboptimal in performance as well as inefficient in terms of parameter sharing. 
In this paper, we delve on research to continually learn user representations task by task, whereby new tasks are learned while using partial parameters from old ones. A new problem arises since when new tasks are trained, previously learned parameters are very likely to be modified, and as a result, an artificial neural network (ANN)-based model may lose its capacity to serve for well-trained previous tasks forever, this issue is termed catastrophic forgetting. To address this issue, we present \emph{Conure} the first \underline{con}tinual, or lifelong, \underline{u}ser \underline{re}presentation learner -- i.e., learning new tasks over time without forgetting old ones. Specifically, we propose iteratively removing less important weights of old tasks in a deep user representation model, motivated by the fact that neural network models are usually over-parameterized. In this way, we could learn many tasks with a single model by reusing the important weights, and modifying the less important weights to adapt to new tasks. We conduct extensive experiments on two real-world datasets with nine tasks and show that \emph{Conure} largely exceeds the standard model that does not purposely preserve such old "knowledge", and performs competitively or sometimes better than models which are trained either individually for each task or simultaneously by merging all task data.


3 Learning Transferable User Representations with Sequential Behaviors via Contrastive Pre-training ICDM2021 https://fajieyuan.github.io/papers/ICDM2021.pdf

Keywords: contrative learnng, self-supervise learning, transfer learning,  pretraining, finetuning, user representation, user profile prediction, cold-start problem

4 User-specific Adaptive Fine-tuning for Cross-domain Recommendations TKDE2021 https://arxiv.org/pdf/2102.10989.pdf

Keywords: adaptive finetuning, pretraining, cold-start problem, cross-domain recommendation

We have also released large-scale dataset (over 1 million user clicking behaviors) for performing transfer learning of user preference in recommendation field
https://github.com/fajieyuan/recommendation_dataset_pretraining
