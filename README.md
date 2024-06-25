# 2024-ml-project-resnet-transfer-learning

Remaining general team-work TODO:
- create Trello or Jira table with TODO tasks so that everyone can assign themselves to something
- the code is mostly done so that our team can focus on data science instead of engineering. I understand you may not enjoy coding but then the other thing that we need to do is data science so I hope you'll be able to draw good conclusions in that area and document them.
- You guys are interns so your job should be to not get stuck on a few lines of code for dozens of hours. So you don't have to implement everything from scratch now and can instead focus on trying to notice areas for improvement in the existing code, interpret the code, describe the code in our /docs directory or 
- contribute by figuring out how to lend a big cloud GPU and run a notebook there - or perhaps we dont need a notebook and can just run our .py files by directly cloning the repo, with the addition of a script that will download the dataset from gdrive - ask GPT for help if you have issues from reading stack overflow or documentation
- generate plantUML diagrams of our model architecture

Data science TODO:
- run the model multiple times to find optimal parameters for best result
- manipulate config.yaml parameters and run each setting separately to end up with a clear description of how each augementation affects the model accuracy / inference probability
- save the best models checkpoints
- add more hyperparameter tuning 
- leverage validation monitoring to potentially implement early stopping
- add more Ensemble Learning methods
    - Bagging (Bootstrap Aggregation)
    - Boosting
    - Stacking

Documentation TODO:
- complete documentation with descriptions of techniques used, problems they solved, our hypotheses and how the results speak to those hypotheses
- then we'll be able to do the presentation - which would focus on pitching our models performance
- improve diagrams such as resnet architecture diagram to be more detailed
- inference diagrams

Utility TODO:
- delete obsolete imports (while familiarizing with repo code)
- add minor improvements such as add any useful prints eg before training, before testing that will let the user know that some process has started, otherwise the console just remains silent for several hours until some process ends and does its prints

Major coding tasks:
- add config.yaml parameter for stuff like balanced_dataset: false, because now it is always enabled by default therefore we won't see how it affects the reuslt and also it takes like 10 minutes

Additional things we could do
- see how our model generalizes on data from a different distribution - create a dev set from images from the internet that are from a different dataset of pneumonia
- use GAN to generate more images
...


Question:
- do I understand it right that inference.py should implement the pipeline by loading a fine tuned model from checkpoints and predicting on a single image? or the purpose of infer in requirements is to incorporate infer function into the training or testing loop?
- should inference do a mean prediction of all ensemble models, or should the ensemble model somehow be a single model loaded from single checkpoint that somehow merges multiple fine tuned models?
- can model performance be evaluated by 1 metric even tho we calculate 5?
