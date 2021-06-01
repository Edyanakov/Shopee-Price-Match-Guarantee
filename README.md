# Shopee-Price-Match-Guarantee
Here I want to share my soultion for Shopee-Price-Match-Guarantee kaggle competition, which secured me 59 public LB and 64 private LB place. </br>
The main challenge was to determine if two products belong to the same category(11014 in total) by their images and text descriptions. </br>
To achive that image model and text model were trained using ArcMarginProduct head. While inference its' feature embeddings were extarcted and similar products were found using cosine similarity with certain threshold.

All data - https://www.kaggle.com/c/shopee-product-matching/data - can be installed via kaggle api. </br>
shopee_inference.ipynb - python notebook used for inference. </br>
shopee_train_bert.ipynb  - python notebook used for training bert models. </br>
shopee_train_image.ipynb  - python notebook used for training image models. </br>
All training was done using kaggle enviroment and Google Colab.

## Bert model
* Paraphrase-xlm-r-multilingual-v1 model from Sentence-Transformers - https://www.sbert.net/index.html
* ArcMargin head with scale 30 and margin 0.5
* AdamW optimizer
* Cosine scheduler with warmup
* Cross entropy loss

## Image model
* Eca_nfnet_l0 model
* Image resized to 544 and center cropped to 512
* Horizontal flip augmentation
* ArcMargin head with scale 30 and margin 0.5
* Adam optimizer or Ranger optimizer with Mish activation function (same validation score)
* Cross entropy loss with normilized weights

## Inference
* Additionaly apply tf-idf model on text
* Translate indonesian text into english
* Ensemble two image models trained on two different folds by concatenating embeddings
* If product have no pair after initial search, increase the threshold and repeat the scan
* Filter some tf-idf and bert predictions if their image embeddings are far off from image model embeddings
* Combine predictions of three separate models by taking only unique ones

## What didnt work
* ArcFace with dynamic margins - https://www.kaggle.com/c/landmark-recognition-2020/discussion/187757
* ArcFace with scale parameter determined by AdaCos - https://www.kaggle.com/c/landmark-retrieval-2020/discussion/176037
* CLIP models - https://github.com/openai/CLIP
* Re-Ranker Cross-Encoder - https://www.sbert.net/examples/applications/retrieve_rerank/README.html
* Focal loss and Triplet loss
* Applying additional layers on top of extracted embeddings 
* Concatenating image embeddings and text embeddings 
* Multiple backbones for image models
* Multiple backbones for bert models
* Bunch of other small things

