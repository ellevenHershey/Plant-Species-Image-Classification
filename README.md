# Plant Species Image Classification - Nightshade Family (Solanaceae) Using Google Teachable Machine

## Project Overview
This project presents a Supervised Learning multi-class image classification model designed to identify and classify 20 distinct plant species from leaf and plant imagery. The model was developed using Google Teachable Machine, a web-based tool that enables the creation of machine learning models through an accessible,drag and drop interface.

The main goal of this project is to streamline plant species identification using visual input. When provided with a plant image, the trained model generates a prediction of the species class together with a confidence score. This system offers valuable applications in botanical research, biodiversity monitoring, agricultural diagnostics, and educational environments where quick and accurate plant identification is essential.

The dataset consists of 5,000 images across 20 plant species classes, with each class containing 250 labeled samples to provide balanced representation for training and evaluation. The model was fully trained, tested, and validated within the Google Teachable Machine platform.

## Plant Species
In this section you will see the 20 nightshade family plant species used to train this model.

### Angels Trumpet

Common Name: Angel’s Trumpet  
Scientific Name: Brugmansia suaveolens  
Description: Brugmansia is a genus in the family Solanaceae, native to South America. Known for its large, pendulous trumpet-shaped flowers, it is highly fragrant but toxic. Ecologically, it attracts nocturnal pollinators. Culturally, it has been used in shamanic rituals due to its hallucinogenic alkaloids, though it is dangerous. Economically, it is valued as an ornamental plant.

### Bellpepper Plant

Common Name: Bell Pepper  
Scientific Name: Capsicum annuum  
Description: Capsicum belongs to the family Solanaceae, native to Central and South America. C. annuum is one of the most economically important vegetables worldwide, producing colorful fruits rich in vitamins. Ecologically, it supports pollinators, while culturally it is a staple in cuisines globally.

### Black Nightshade Plant

Common Name: Black Nightshade  
Scientific Name: Solanum nigrum  
Description: Solanum is a large genus in the family Solanaceae. S. nigrum is an annual herb found worldwide, thriving in disturbed soils. Ecologically, it is a hardy weed. Culturally, some varieties are consumed as leafy vegetables or used medicinally, though others are toxic.

### Cape Gooseberry Plant

Common Name: Cape Gooseberry / Goldenberry  
Scientific Name: Physalis peruviana  
Description: Physalis (family Solanaceae) produces fruits enclosed in papery husks. P. peruviana is native to South America and cultivated for jams, juices, and export. Ecologically, its husk protects fruit from pests. Economically, it is marketed as a “superfruit” for its antioxidants.

### Chayote Plant

Common Name: Chayote / Sayote  
Scientific Name: Sechium edule    
Description: Sechium belongs to the family Cucurbitaceae, native to Mesoamerica. S. edule is a perennial vine producing pear-shaped fruits. Ecologically, it grows vigorously in tropical climates. Economically, it is a staple vegetable in Latin America and Asia, with edible shoots and tubers.

### Chili Pepper Plant
Common Name: Chili Pepper  
Scientific Name: Capsicum annuum (and related species)  
Description: Capsicum (family Solanaceae) includes pungent fruits rich in capsaicin. Ecologically, capsaicin deters herbivores. Culturally, chilies are central to cuisines worldwide and used medicinally for circulation. Economically, they are a major spice crop.

### Cocona Plant
Common Name: Cocona  
Scientific Name: Solanum sessiliflorum  
Description: Solanum (family Solanaceae) includes tropical shrubs. S. sessiliflorum, native to the Amazon, produces orange fruits used in juices and medicine. Ecologically, it thrives in humid forests. Economically, it is locally cultivated but less widespread than naranjilla.

### Common Desert Thorn Plant
Common Name: Desert Thorn  
Scientific Name: Lycium shawii  
Description: Lycium is a genus in the family Solanaceae, adapted to arid regions. L. shawii is native to deserts of the Middle East and Africa. Ecologically, it stabilizes soils and provides food for wildlife. Culturally, it has medicinal uses in traditional remedies.

### Deadly Nightshade Plant

Common Name: Deadly Nightshade / Belladonna
Scientific Name: Atropa belladonna
Description: Atropa (family Solanaceae) is infamous for toxic alkaloids. A. belladonna is a perennial herb with violet flowers and black berries. Ecologically, it deters herbivores. Historically, it was used in cosmetics and medicine despite its danger.

### Eggplant Plant

Common Name: Eggplant / Brinjal  
Scientific Name: Solanum melongena  
Description: Solanum (family Solanaceae) includes staple crops. S. melongena is native to Southeast Asia, producing purple fruits. Ecologically, it supports pollinators. Economically, it is a major vegetable in Mediterranean and Asian cuisines.

### Jerusalem Cherry Plant

Common Name: Jerusalem Cherry  
Scientific Name: Solanum pseudocapsicum  
Description: Solanum (family Solanaceae) includes ornamental shrubs. S. pseudocapsicum produces bright red berries, mildly toxic. Ecologically, it is hardy in varied climates. Economically, it is grown indoors as a decorative plant.

### Jimsonweed Plant

Common Name: Jimsonweed / Devil’s Snare  
Scientific Name: Datura stramonium  
Description: Datura (family Solanaceae) is known for hallucinogenic alkaloids. D. stramonium is an invasive annual herb with trumpet flowers. Ecologically, it colonizes disturbed soils. Culturally, it has ritual uses but is highly toxic.

### Naranjilla Plant

Common Name: Naranjilla / Lulo
Scientific Name: Solanum quitoense
Description: Solanum (family Solanaceae) includes tropical shrubs. S. quitoense, native to the Andes, produces spiny leaves and orange fruits. Economically, it is cultivated for juices. Ecologically, it thrives in humid highlands.

### Pepino Plant

Common Name: Pepino / Melon Pear  
Scientific Name: Solanum muricatum  
Description: Solanum (family Solanaceae) includes subtropical shrubs. S. muricatum produces sweet, melon-like fruits. Economically, it is cultivated in the Andes and exported. Ecologically, it grows in temperate climates.

### Petunia Plant

Common Name: Petunia  
Scientific Name: Petunia × hybrida  
Description: Petunia (family Solanaceae) is native to South America. P. × hybrida is a hybrid ornamental with trumpet-shaped flowers. Economically, it is a popular garden plant. Ecologically, it attracts pollinators.

### Red Buffalo Bur Plant

Common Name: Red Buffalo-bur  
Scientific Name: Solanum sisymbriifolium  
Description: Solanum (family Solanaceae) includes spiny nightshades. S. sisymbriifolium produces bright red fruits. Ecologically, it is invasive. Economically, it is studied for pest resistance in crops.

### Silverleaf Nightshade Plant

Common Name: Silverleaf Nightshade  
Scientific Name: Solanum elaeagnifolium  
Description: Solanum (family Solanaceae) includes hardy perennials. S. elaeagnifolium has silvery leaves and purple flowers. Ecologically, it is drought-tolerant. Economically, it is considered a weed but used medicinally.

### Tobacco Plant

Common Name: Tobacco  
Scientific Name: Nicotiana tabacum  
Description: Nicotiana (family Solanaceae) is native to the Americas. N. tabacum is cultivated for its leaves, processed into tobacco products. Economically, it is globally significant. Ecologically, it supports pollinators but is chemically defended.

### Tree Tomato Plant

Common Name: Tree Tomato / Tamarillo  
Scientific Name: Solanum betaceum  
Description: Solanum (family Solanaceae) includes subtropical shrubs. S. betaceum, native to the Andes, produces tangy fruits. Economically, it is cultivated for juices. Ecologically, it thrives in highland climates.

### Wolfberry Plant

Common Name: Wolfberry / Goji Berry  
Scientific Name: Lycium barbarum / Lycium chinense  
Description: Lycium (family Solanaceae) includes shrubs native to East Asia. These species produce red berries rich in antioxidants. Economically, they are marketed as health foods. Ecologically, they adapt to varied soils and climates.



## Model Training details 
In this section you will see the epochs, batch size, learning rate, number of images per size, and tests made. 

### Epochs        = 351
### Batch Size    = 256
### Learning Rate = 0.0011

### Dataset Distribution 
| No. | Plant Species        | Number of Images |
|:---:|:--------------------:|:----------------:|
|  1  | Angels' Trumpet      | 250              |
|  2  | Bell Pepper          | 250              |
|  3  | Black Nightshade     | 250              |
|  4  | Cape Gooseberry      | 250              |
|  5  | Chayote              | 250              |
|  6  |    Chili Pepper      | 250              |
|  7  | Cocona               | 250              |
|  8  | Common Desert Thorn  | 250              |
|  9  | Deadly Nightshade    | 250              |
|  10 | Eggplant             | 250              |
|  11 | Jerusalem  Cherry    | 250              |
|  12 | Jimsonweed           | 250              |
|  13 | Naranjilla           | 250              |
|  14 | Pepino               | 250              |
|  15 | Petunia              | 250              |
|  16 | Red Buffalo Bur      | 250              |
|  17 | Silverleaf Nightshade| 250              |
|  18 | Tobacco              | 250              |
|  19 | Tree Tomato          | 250              |
|  20 | Wolfberry            | 250              |
|     |               Total: | 5,000


## Model Evaluation 

### Confusion Matrix 

### Accuracy per Class

### Overall Model Accuracy
#### Training Accuracy

#### Training Loss


## Model Testing
In this section, the model's generalization capability and accuracy prediction are validatet through 10 independent test cases. Each case involves a previously unseen image, with the calssification output recorded, including the predicted species and its confidence percentage. These test cases provide observed evidence of the model's inference performance beyond the training and validation datasets
