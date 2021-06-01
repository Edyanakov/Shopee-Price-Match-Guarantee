# Shopee-Price-Match-Guarantee
Here I want to share my soultion for Shopee-Price-Match-Guarantee kaggle competition, which secured me 59 public LB and 64 private LB place. </br>
The main challenge was to determine if two products belong to the same category(11014 in total) by their images and text descriptions.

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
