# GitHub Kaggle Project README

---

### **👥 Team Members**

| Name | GitHub Handle | Contribution |
| ----- | ----- | ----- |
| David Wang | @davidwang5 | Worked on CNN model, performed data augmentation |
| Kristal Hong | @khong707 | Worked on CNN model, performed data augmentation |
| Suzette Mejia | @suzettem16 | Worked on CNN model, performed data augmentation |
| Stella Huang | @stellahuang0527 | Worked on CNN model, performed data augmentation |


---

## **🎯 Project Highlights**

**Example:**

* Built a CNN-based deep learning model with transfer learning and fine-tuning to solve to classify dermatological conditions across diverse skin tones
* Achieved a ranking of 37 out of 73 on the Kaggle Leaderboard
* Used Grad-CAM to interpret model decisions to interpret model decisions
* Implemented image normalization to optimize results within compute constraints

🔗 [Equitable AI for Dermatology | Kaggle Competition Page](https://www.kaggle.com/competitions/bttai-ajl-2025/overview)
🔗 [WiDS Datathon 2025 | Kaggle Competition Page](https://www.kaggle.com/competitions/widsdatathon2025/overview)

---

## **👩🏽‍💻 Setup & Execution**

* To reproduce this project, first clone the repository using git clone https://github.com/your-username/dermatology-ai. Then, set up the environment by creating a virtual environment (e.g., using python -m venv env) and activating it.

* This project is designed to run in Kaggle Notebooks, so no local dataset downloads are necessary. 

* The dataset is automatically available via the competition page at /kaggle/input/bttai-ajl-2025/ once you join the Break Through Tech x Algorithmic Justice League competition on Kaggle.

---

## **🏗️ Project Overview**

* This project was developed as part of the Break Through Tech AI Program in collaboration with the Algorithmic Justice League through a Kaggle-hosted competition. The competition challenges participants to build inclusive machine learning models capable of classifying dermatological conditions from images across diverse skin tones. 
 * The primary objective is to address bias in dermatology AI tools, which historically underperform for people with darker skin due to underrepresentation in training data.
 * By creating a fairer and more accurate classification model, our work has the potential to reduce diagnostic errors, improve early detection, and promote equitable healthcare outcomes for marginalized communities. This project not only advances technical performance in medical AI but also contributes to broader efforts in algorithmic fairness and social impact.

---

## **📊 Data Exploration**

* Dataset: The dataset we used is a subset of the FitzPatrick17k dataset. This is a labeled collection of around 17,000 images that depict serious and cosmetic dermatological conditions such as melanoma and acne among a range of skin tones scored on the FitzPatrick skin tone scale. We took a sample of around 4,500 images from this dataset, which represented 21 skin conditions out of over 100.

* Data Exploration and Preprocessing: To begin exploring our data, we viewed the variety of images that contained skin conditions, as outlined by the "Evaluating Deep Neural Networks Trained on Skin Images with the Fitzpatrick 17k Dataset" video provided to us. We also examined the metadata provided to us as well as conducted our own research on how machine learning is currently being tested in real clinical settings to help with determining and treating dermatalogical conditions.

* After exploring the data and performing filename adjustments such as constructing file paths, we started data preprocessing beginning with using scikit-learn’s LabelEncoder to transform string labels into integers. We split the data into training and validation sets and used the ImageDataGenerator to preprocess image data by rescaling pixel values from their original range (0–255) to a normalized range. We used the helper function create_generator to generate batches of image data directly from the dataframe, and then were able to train a Keras Sequential CNN model.
  
* Challenges we faced: A major challenge that we faced when working with this dataset for our Kaggle competition was the fact that we needed to work with image data using convolutional neural networks instead of the traditional deep neural networks that we were used to using. In addition, we made a few assumptions about our data, in particular assuming that all images can be uniformly resized to the target size without losing critical information.


Visualizations of preprocessing steps and Model Accuracy/Loss charts are listed in this Github repository as Loss and Accuracy.png and Skin_Tone_Across_Classes.png.


---

## **🧠 Model Development**

**Describe (as applicable):**
We used two different models to make progress in achieving our primary goal of increasing classification accuracy among all skin tones. The first model was CNN... The second model was the pre-trained model EfficientNetB0...
* Model(s) used (e.g., CNN with transfer learning, regression models)
* Feature selection and Hyperparameter tuning strategies
* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)

---

## **📈 Results & Key Findings**

**Describe (as applicable):**

* Performance metrics (e.g., Kaggle Leaderboard score, F1-score)
* How your model performed overall
* How your model performed across different skin tones (AJL)
* Insights from evaluating model fairness (AJL)

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## **🖼️ Impact Narrative**

**Answer the relevant questions below based on your competition:**

**WiDS challenge:**

1. What brain activity patterns are associated with ADHD; are they different between males and females, and, if so, how?
2. How could your work help contribute to ADHD research and/or clinical care?

**AJL challenge:**

As Dr. Randi mentioned in her challenge overview, “Through poetry, art, and storytelling, you can reach others who might not know enough to understand what’s happening with the machine learning model or data visualizations, but might still be heavily impacted by this kind of work.”
As you answer the questions below, consider using not only text, but also illustrations, annotated visualizations, poetry, or other creative techniques to make your work accessible to a wider audience.
Check out [this guide](https://drive.google.com/file/d/1kYKaVNR\_l7Abx2kebs3AdDi6TlPviC3q/view) from the Algorithmic Justice League for inspiration!

1. What steps did you take to address [model fairness](https://haas.berkeley.edu/wp-content/uploads/What-is-fairness_-EGAL2.pdf)? (e.g., leveraging data augmentation techniques to account for training dataset imbalances; using a validation set to assess model performance across different skin tones)
2. What broader impact could your work have?

---

## **🚀 Next Steps & Future Improvements**

**Address the following:**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

---

## **📄 References & Additional Resources**

* Cite any relevant papers, articles, or tools used in your project

---

