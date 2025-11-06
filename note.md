## Prepare Image
1. Extract Regions of Interest (ROIs): They use Faster R-CNN (pre-trained on Visual Genome) to extract 36 high-confidence regions from each image.
2. Obtain Visual Features: for each region, extract 2048-dimensional visual features (i.e., a vector representing the visual content of that region). These features are then projected into the same embedding space as BART’s text tokens (dimension 768) using a linear layer.
3. Export feature into .npy and .npz

## Prepare Text
Replace the aspect terms with special token `$T$`, and set these aspect terms as label for predicting

### Format

* We provide two kinds of format, one is ".txt" for LSTM-based models, and another is "tsv" for BERT models.

* For example, each row of "train.tsv" for twitter2015 is one sample:
  (1). the first column is index;
  (2). the second column is sentiment label (0 refers to negative, 1 refers to neutral, and 2 refers to positive);
  (3). the third column is the id for the corresponding image of this tweet, which can be found in the folder "twitter2015_images";
  (4). the fourth and fifth columns respectively refer to the original tweet by masking the current opinion target and the opinion target (i.e., entity).
  
  Note that each tweet may contain multiple opinion targets (i.e., entities), it may correspond to several continuous samples. E.g., the first and second samples in "train.tsv" for twitter2015 are about the same tweet but different entities.
  
 * The ".txt" file is similar to "train.tsv", but every four lines in the file is one sample:
  (1). the first line refers to the original tweet by masking the current opinion target;
  (2). the second line refers to  the opinion target (i.e., entity);
  (3). the third line is sentiment label (Note that here -1 refers to negative, 0 refers to neutral, and 1 refers to positive);
  (4). the fourth line is the id for the corresponding image of this tweet, which can be found in the folder "twitter2015_images".