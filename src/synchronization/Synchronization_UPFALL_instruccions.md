# UPFALL Dataset Synchronization Instructions

To use UPFALL Dataset in this library, you have to do the following steps:

Script to prepare data of the UP-Fall (Har-up) dataset from the official website: https://sites.google.com/up.edu.mx/har-up/

1. The user must download the Camera1 and Camera2 zip files from the Downloads section.
2. The sensor csv must be downloaded (https://drive.google.com/file/d/1JBGU5W2uq9rl8h7bJNt2lN4SjfZnFxmQ/view). 
   The folder structure must be as the following:

```
BodyFlow
│   README.md
|   ...    
│─── upfall
│   │─── Subject1
│   │    │─── Activity1
│   │       │─── Trial1
│   │           │─── Subject1Activity1Trial1Camera1.zip
│   │           │─── Subject1Activity1Trial1Camera2.zip
│   ...
│   │─── SubjectN
│   │       │─── ActivityN
│   │          │─── TrialN
│   │               │─── SubjectNActivityNTrialNCamera1.zip
│   │               │─── SubjectNActivityNTrialNCamera2.zip
│   │
│   │─── CompleteDataSet.csv
```

Output:

* A single file 'processed_upfall.csv' with the synchronized data. 
* Two text files: 'missing_filles.txt' - missing files.
                  'processing_list.txt' - processing list. 
* File in 'logs' folder with all the processed output files.
"""