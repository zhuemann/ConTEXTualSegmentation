# ConTEXTual Net: A Multimodal Vision-Language Model for Segmentation of Pneumothorax
ConTEXTual net is a multimodal vision-langauge segmentation model for medical imaging. This github repo is still under construction for open source use. 
## Data
Information on the CANDID-PTX dataset can be found in their paper found here: https://pubs.rsna.org/doi/10.1148/ryai.2021210136
And the data can be accessed after signing a data use agreement and an online ethics course: https://pubs.rsna.org/doi/10.1148/ryai.2021210136
Once you have downloaded the data, unzip all of the files and move them to the same folder with the file stucture below.
```
candid_ptx
|   Pneumothorax_reports.csv
|   chest_tube.csv
|   acute_rib_fracture.csv
|
|___chest_radiographs
    |   *images

```
Note that the image_name_text_label_redacted.csv file does not contain the text or labels due to data use restrictions. Users will have to create this by joining the csv from the source dataset with the cases postive for pnuemothorax. Included are the images used in the work. 
## Usage
