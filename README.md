# coffee_bean_image_classification
I completed this project for an image processing class for my graduate degree in Winter 2026. It uses image processing techniques to classify images of coffee beans into distinct roast levels.

The data comes from [Kaggle](https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224/data) and consists of 1,194 unique images, or about 400 images per distinct roast level: unroasted (green), light roast, medium roast, and dark roast.

The pipeline consists of preprocessing (background removal, HSV-based refinement, and morphological operations), extraction of 15 features from bean pixels, and a Random Forest classifier. On a held-out test set, the model achieved 97% accuracy with strong per-class precision and recall. Batch-level consistency analysis was evaluated on synthetic batches, achieving expected consistency scores for each. Evaluation on custom images was also performed. The approach shows that classical image processing combined with a tuned Random Forest is effective for this task.

**Data and preprocessing**:

The primary dataset is the Coffee Bean Dataset Resized (224x224) from Kaggle, with 400 images per roast level (Green, Light, Medium, Dark) originally split into train and test. The preprocessing notebook merges train and test into single per-roast folders, then identifies duplicate images by file hash and removes duplicates, yielding 1,194 unique images. Background removal is applied to each image and the result is saved in place. The removal pipeline uses: (1) Otsu’s method on the grayscale image to obtain an initial bean mask; (2) HSV-based rules to exclude bright low-saturation regions (background) and dark low-saturation regions (shadows); (3) hole-filling (scipy.ndimage.binary_fill_holes) and morphological close and open with an elliptical kernel (OpenCV) to produce a clean foreground mask. Only pixels inside this mask are used for feature extraction. 

**Feature Extraction:**

Features are computed only over the bean pixels (masked region). The 15 features are: mean and standard deviation of R, G, and B; mean and standard deviation of H, S, and V; ratios R/G and B/G; and the standard deviation of grayscale intensity over the bean (gray_std). These capture color distribution, color space, and a simple texture-related measure. The ratios of R/G and B/G pixels provided the best separation between classes of any features.

**Classification and hyperparameter tuning:**

A scikit-learn Pipeline couples StandardScaler and RandomForestClassifier. RandomizedSearchCV searches over the number of trees (50, 100, 200), max depth (None, 10, 20, 30), min_samples_split (2, 5, 10), and min_samples_leaf (1, 2, 4), with 24 random combinations, 5-fold stratified cross-validation, and accuracy as the scoring metric. The best configuration is refit on the full training set. The data are split once: 80% train (955 images), 20% test (239 images), stratified by roast level. All preprocessing and feature extraction were implemented in Python (OpenCV, NumPy, SciPy); classification and tuning use scikit-learn and joblib for saving the model, scaler, and metadata.

RandomizedSearchCV selected the following best configuration (on the training set via 5-fold CV): n_estimators = 50, min_samples_split = 2, min_samples_leaf = 1, max_depth = 30. Best cross-validation accuracy was 0.9812. The same pipeline evaluated with 5-fold cross-validation on the full training set yielded a mean accuracy of 0.9801.

**Evaluation:**

Test-set metrics (accuracy, confusion matrix, precision, recall, F1) are computed on the held-out 239 images. Custom images are resized to 224x224 pixels, passed through the same background-removal and feature-extraction pipeline, and then scored with the saved model; accuracy and confusion matrix are reported for the labeled custom set. Batch consistency is evaluated by constructing three synthetic batches from the roast-level folders: a “perfect” batch (100 images from one roast), a “good” batch (80 from one roast and 20 from adjacent roasts, true label = majority roast), and a “bad” batch (25 images from each of the four roasts, true label = majority roast). For each batch, the proportion of beans predicted as the batch’s intended roast is reported as the consistency score.

On the held-out test set of 239 images, the model achieved overall accuracy of 97.07%. The confusion matrix (rows = true label, columns = predicted) is:

|            | **Green** | **Light** | **Medium** | **Dark** |
|------------|-----------|-----------|------------|----------|
| **Green**  | 60        | 0         | 0          | 0        |
| **Light**  | 0         | 59        | 1          | 0        |
| **Medium** | 0         | 1         | 58         | 0        |
| **Dark**   | 0         | 0         | 4          | 55       |

Per-class precision, recall, and F1 (macro-averaged) are high: Green (precision 0.98, recall 0.93), Light (1.00, 1.00), Medium (0.98, 0.98), Dark (0.92, 0.97). Green is never confused with other classes; confusions occur mainly between adjacent roasts (Light–Medium, Medium–Dark). A baseline linear regression (ordinal encoding of roast level) achieved approximately 58% accuracy on the same test set, so the Random Forest improves performance by a large margin.

A set of custom images of coffee beans (photographed by me with coffee I had at home) was labeled manually and evaluated. Images were resized to 224x224, preprocessed with the same background-removal pipeline, and passed through the trained model. Accuracy on the 20 labeled custom images was 65%; the model tended to confuse Medium with Dark, as the medium-roast beans used were on the darker side.

**Limitations and opportunities for improvement:**

Some confusion remains between adjacent roasts (such as Light and Medium, or Medium and Dark), which is consistent with the visual similarity of neighboring roast levels. Custom-image accuracy may be lower than test-set accuracy due to different lighting, background, or camera; this should be reported explicitly when the custom set is finalized. Texture is represented only by variance (such as gray_std and channel standard deviations), not by dedicated descriptors. Batch consistency was evaluated on synthetic batches built from the same dataset; real-world batches would require separate data collection.

Adding more explicit texture features could improve discrimination where color is ambiguous. Comparing different color spaces or feature subsets in a formal experiment would strengthen the methodology section. Trying other classifiers (such as SVM or gradient boosting) or calibrating probabilities could be useful for applications that need confidence scores. Collecting real batch data (multiple beans per batch with a batch-level label) would allow validation of consistency in the field. Data augmentation or lighting normalization could improve robustness on custom or in-the-wild images, such as those taken on backgrounds of other colors.
