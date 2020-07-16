from fastai.vision import *
from fastai.widgets import*
import numpy as np

classes = ['ac','as','cb','cc','ci','cs','cuCon','cu','ns','sc','st']

# %%
path = Path('images/')
#for name in classes:
#    folder = name
#    file = name + '.csv'
#    dest = path/folder
#    dest.mkdir(parents = True, exist_ok = True)
#    download_images(path/file, dest, max_pics = 1000)


# %%

for c in classes:
    print(c)
    verify_images(path/c, delete = True, max_size = 500)
    
# %%

np.random.seed()
data = ImageDataBunch.from_folder(path, train = ".", valid_pct = 0.2, ds_tfms = get_transforms(), size = 224, num_workers = 4).normalize(imagenet_stats)

# %%

data.classes
data.show_batch(rows = 3, figsize = (7, 8))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)

# %%
learn = cnn_learner(data, models.resnet34, metrics = error_rate)
learn.fit_one_cycle(50)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()

# %%
learn.fit_one_cycle(50, max_lr = slice(1e-6, 1e-4))
learn.save('stage-2')

# %%
learn.load('stage-2')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

# %% Testing on new images
learn.export()
defaults.device = torch.device('cpu')

# %%
img = open_image('test/test5.jpg')
learn = load_learner(path)
pred_class,pred_idx,outputs = learn.predict(img)
pred_class.obj