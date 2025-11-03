from datasets import load_dataset
ds = load_dataset("glue", "sst2")              # shows splits and counts

for key in ds.keys():
    ##print(ds[key][0])
    print(ds[key]['idx'])
    

##print(ds["train"].column_names)        # e.g. ['sentence', 'label']
##print(ds["train"].features)            # shows feature types (ClassLabel etc.)
##print(len(ds["train"]), len(ds["validation"]), len(ds["test"]))
