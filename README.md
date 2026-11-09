# dr-early-detection
Code for 'Development of an Intelligent System for Early Detection of Diabetic Retinopathy Using Retinal Images'.
Run order: prep.py -> hash2.py -> meta.py 10 -> lesion_stats.py -> extract.py 224 idrid ddr aptos eyepacs -> train.py GATE 0; train.py B1 0; train.py P 0 (EPOCHS=5) -> evaluate.py -> fig*.py
splits/meta.csv: image list, patient split, exclusion flags. License: MIT.
