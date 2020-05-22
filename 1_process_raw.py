# %%
import pandas as pd
import numpy as np
import json
from glob import glob
from smoothdoq.processing import parse_raw_counts
from tqdm import tqdm
import os

MIN_COUNTS_TO_BINS_RATIO = 2
MIN_BINS = 1
MIN_COUNTS = 1

# %%
names = ['obj', 'head', 'dim', 'mean', 'perc_5', 
         'perc_25', 'median', 'perc_75', 'perc_95', 
         'std', 'raw'] 
files = sorted(glob("./data/raw/doq_raw*.csv"))

if os.path.exists("data/ignored2.tsv"):
    os.remove("data/ignored2.tsv")
ignored = open("data/ignored2.tsv", "a")
ignored.write("obj\tdim\tcounts\tbins\tfilenum\n")

if os.path.exists("data/flagged2.tsv"):
    os.remove("data/flagged2.tsv")
flagged = open("data/flagged2.tsv", "a")
flagged.write("obj\tdim\tflags\tbins\tcounts\n")

# %%
hard_list = {('spaceship', 'SPEED'), ('caravan', 'SPEED'), ('chapel', 'LENGTH'), ('wolf', 'SPEED'), ('canola', 'MASS'), ('mustang', 'SPEED'), ('crane', 'SPEED'), ('appartment', 'LENGTH'), ('dinghy', 'SPEED'), ('thyme', 'MASS'), ('bmw', 'SPEED'), ('caf\\u00e9', 'LENGTH'), ('motorboat', 'SPEED'), ('pushing', 'SPEED'), ('squid', 'SPEED'), ('ssd', 'MASS'), ('mansion', 'LENGTH'), ('retina', 'LENGTH'), ('airplane', 'SPEED'), ('toy', 'SPEED'), ('bean', 'MASS'), ('dashing', 'SPEED'), ('tomato', 'MASS'), ('insect', 'SPEED'), ('deer', 'SPEED'), ('strolling', 'SPEED'), ('clover', 'MASS'), ('sorghum', 'MASS'), ('ambling', 'SPEED'), ('scallion', 'MASS'), ('spider', 'MASS'), ('barn', 'LENGTH'), ('ipod', 'MASS'), ('duck', 'SPEED'), ('pea', 'MASS'), ('camera', 'MASS'), ('computer', 'MASS'), ('carp', 'SPEED'), ('strawberry', 'MASS'), ('palm', 'MASS'), ('sailboat', 'SPEED'), ('sesame', 'MASS'), ('lemur', 'SPEED'), ('ginger', 'MASS'), ('router', 'MASS'), ('sprinting', 'SPEED'), ('papaya', 'MASS'), ('pepper', 'MASS'), ('earphone', 'MASS'), ('lemur', 'MASS'), ('onion', 'MASS'), ('watermelon', 'MASS'), ('palm', 'LENGTH'), ('whizzing', 'SPEED'), ('strutting', 'SPEED'), ('skunk', 'MASS'), ('grape', 'MASS'), ('snowmobile', 'SPEED'), ('traversing', 'SPEED'), ('dill', 'MASS'), ('snake', 'SPEED'), ('crab', 'SPEED'), ('joystick', 'MASS'), ('leopard', 'SPEED'), ('romaine', 'MASS'), ('trudging', 'SPEED'), ('parrot', 'MASS'), ('steamboat', 'SPEED'), ('moose', 'MASS'), ('blueberry', 'MASS'), ('basil', 'MASS'), ('disk', 'MASS'), ('coupe', 'SPEED'), ('taxi', 'SPEED'), ('lettuce', 'MASS'), ('tortoise', 'SPEED'), ('transmitter', 'MASS'), ('sleigh', 'SPEED'), ('trolley', 'SPEED'), ('toy', 'MASS'), ('bird', 'MASS'), ('monastery', 'LENGTH'), ('office', 'LENGTH'), ('bat', 'SPEED'), ('daikon', 'MASS'), ('cotton', 'MASS'), ('kicking', 'SPEED'), ('taro', 'MASS'), ('salon', 'LENGTH'), ('gorilla', 'MASS'), ('sheep', 'MASS'), ('radish', 'MASS'), ('riding', 'SPEED'), ('jet', 'SPEED'), ('stomping', 'SPEED'), ('ethernet', 'MASS'), ('wasabi', 'MASS'), ('steamer', 'SPEED'), ('cheetah', 'MASS'), ('tanker', 'SPEED'), ('lemon', 'MASS'), ('snake', 'MASS'), ('handset', 'MASS'), ('raccoon', 'MASS'), ('camcorder', 'MASS'), ('webcam', 'MASS'), ('cheetah', 'SPEED'), ('speaker', 'MASS'), ('stepping', 'SPEED'), ('moping', 'SPEED'), ('bull', 'SPEED'), ('swimming', 'SPEED'), ('snorkelling', 'SPEED'), ('safflower', 'MASS'), ('freighter', 'SPEED'), ('dog', 'SPEED'), ('cacao', 'MASS'), ('alfalfa', 'MASS'), ('hurrying', 'SPEED'), ('turtle', 'MASS'), ('shrimp', 'MASS'), ('nerve', 'LENGTH'), ('rv', 'SPEED'), ('b&b', 'LENGTH'), ('lion', 'MASS'), ('otter', 'MASS'), ('receiver', 'MASS'), ('hurtling', 'SPEED'), ('dolphin', 'SPEED'), ('striding', 'SPEED'), ('whale', 'SPEED'), ('boar', 'SPEED'), ('deli', 'LENGTH'), ('rocket', 'SPEED'), ('perineum', 'LENGTH'), ('shallot', 'MASS'), ('teeming', 'SPEED'), ('sugarcane', 'MASS'), ('snorkeling', 'SPEED'), ('poodle', 'MASS'), ('cowpea', 'MASS'), ('tongue', 'LENGTH'), ('owl', 'SPEED'), ('chicory', 'MASS'), ('excavator', 'SPEED'), ('boutique', 'LENGTH'), ('jalapeno', 'MASS'), ('courtyard', 'LENGTH'), ('robot', 'SPEED'), ('kangaroo', 'MASS'), ('ambulance', 'SPEED'), ('jeep', 'SPEED'), ('cave', 'LENGTH'), ('walking', 'SPEED'), ('turning', 'SPEED'), ('shark', 'SPEED'), ('motorcycle', 'SPEED'), ('jatropha', 'MASS'), ('scooting', 'SPEED'), ('pistachio', 'MASS'), ('cattle', 'SPEED'), ('connector', 'MASS'), ('fox', 'SPEED'), ('iguana', 'MASS'), ('scalp', 'LENGTH'), ('rushing', 'SPEED'), ('cable', 'MASS'), ('sheep', 'SPEED'), ('lorry', 'SPEED'), ('gecko', 'SPEED'), ('trekking', 'SPEED'), ('diving', 'SPEED'), ('galloping', 'SPEED'), ('meandering', 'SPEED'), ('antenna', 'MASS'), ('gliding', 'SPEED'), ('aircraft', 'SPEED'), ('watercraft', 'SPEED'), ('microphone', 'MASS'), ('rhino', 'MASS'), ('marching', 'SPEED'), ('shrimp', 'SPEED'), ('rhino', 'SPEED'), ('otter', 'SPEED'), ('slithering', 'SPEED'), ('minivan', 'SPEED'), ('hummingbird', 'SPEED'), ('arugula', 'MASS'), ('lounge', 'LENGTH'), ('crocodile', 'SPEED'), ('limo', 'SPEED'), ('carriage', 'SPEED'), ('tavern', 'LENGTH'), ('arse', 'LENGTH'), ('pigeon', 'MASS'), ('chard', 'MASS'), ('cherry', 'MASS'), ('trout', 'SPEED'), ('rabbit', 'SPEED'), ('ute', 'SPEED'), ('swinging', 'SPEED'), ('koala', 'SPEED'), ('cane', 'MASS'), ('drifting', 'SPEED'), ('bookstore', 'LENGTH'), ('chicken', 'SPEED'), ('floating', 'SPEED'), ('farm', 'LENGTH'), ('battery', 'MASS'), ('squirrel', 'MASS'), ('plaza', 'LENGTH'), ('boat', 'SPEED'), ('train', 'SPEED'), ('camper', 'SPEED'), ('motorhome', 'SPEED'), ('printer', 'MASS'), ('gecko', 'MASS'), ('pointing', 'SPEED'), ('cilantro', 'MASS'), ('squirrel', 'SPEED'), ('mic', 'MASS'), ('sled', 'SPEED'), ('gorilla', 'SPEED'), ('tarragon', 'MASS'), ('geranium', 'MASS'), ('grapevine', 'MASS'), ('jumping', 'SPEED'), ('leek', 'MASS'), ('rig', 'SPEED'), ('parrot', 'SPEED'), ('airship', 'SPEED'), ('boar', 'MASS'), ('modem', 'MASS'), ('dragonfly', 'MASS'), ('alligator', 'MASS'), ('lavender', 'MASS'), ('wheelchair', 'SPEED'), ('trawling', 'SPEED'), ('schooner', 'SPEED'), ('groundnut', 'MASS'), ('cabin', 'SPEED'), ('goat', 'SPEED'), ('headset', 'MASS'), ('backpacking', 'SPEED'), ('kayak', 'SPEED'), ('capsicum', 'MASS'), ('tortoise', 'MASS'), ('synagogue', 'LENGTH'), ('bull', 'MASS'), ('helicopter', 'SPEED'), ('cantaloupe', 'MASS'), ('adapter', 'MASS'), ('throne', 'LENGTH'), ('turtle', 'SPEED'), ('eatery', 'LENGTH'), ('leaping', 'SPEED'), ('elephant', 'MASS'), ('travelling', 'SPEED'), ('moving', 'SPEED'), ('trailer', 'SPEED'), ('home', 'LENGTH'), ('duplex', 'LENGTH'), ('crab', 'MASS'), ('parsnip', 'MASS'), ('running', 'SPEED'), ('sensor', 'MASS'), ('posing', 'SPEED'), ('poodle', 'SPEED'), ('courthouse', 'LENGTH'), ('skiff', 'SPEED'), ('truck', 'SPEED'), ('laptop', 'MASS'), ('bungalow', 'LENGTH'), ('dongle', 'MASS'), ('lunging', 'SPEED'), ('chalet', 'LENGTH'), ('entrance', 'LENGTH'), ('trolling', 'SPEED'), ('rover', 'SPEED'), ('whale', 'MASS'), ('rabbit', 'MASS'), ('barreling', 'SPEED'), ('vector', 'SPEED'), ('guesthouse', 'LENGTH'), ('camry', 'SPEED'), ('elephant', 'SPEED'), ('telephone', 'MASS'), ('melon', 'MASS'), ('chasing', 'SPEED'), ('ceiling', 'LENGTH'), ('camel', 'MASS'), ('bunny', 'MASS'), ('daybed', 'LENGTH'), ('owl', 'MASS'), ('tablet', 'MASS'), ('bird', 'SPEED'), ('puppy', 'SPEED'), ('crawling', 'SPEED'), ('zooming', 'SPEED'), ('rickshaw', 'SPEED'), ('splashing', 'SPEED'), ('fennel', 'MASS'), ('socket', 'MASS'), ('atv', 'SPEED'), ('bunny', 'SPEED'), ('glider', 'SPEED'), ('avocado', 'MASS'), ('breast', 'LENGTH'), ('flying', 'SPEED'), ('skunk', 'SPEED'), ('airliner', 'SPEED'), ('ship', 'SPEED'), ('raspberry', 'MASS'), ('carrot', 'MASS'), ('driving', 'SPEED'), ('leopard', 'MASS'), ('suv', 'SPEED'), ('hippo', 'MASS'), ('celery', 'MASS'), ('moose', 'SPEED'), ('hummingbird', 'MASS'), ('darting', 'SPEED'), ('orange', 'MASS'), ('convoy', 'SPEED'), ('tiger', 'SPEED'), ('camel', 'SPEED'), ('iguana', 'SPEED'), ('chicken', 'MASS'), ('peanut', 'MASS'), ('wagon', 'SPEED'), ('lion', 'SPEED'), ('hippo', 'SPEED'), ('scorpion', 'SPEED'), ('gallivanting', 'SPEED'), ('frog', 'MASS'), ('parsley', 'MASS'), ('frog', 'SPEED'), ('mint', 'MASS'), ('beet', 'MASS'), ('sage', 'MASS'), ('pear', 'MASS'), ('endive', 'MASS'), ('cruising', 'SPEED'), ('alligator', 'SPEED'), ('blowing', 'SPEED'), ('headphone', 'MASS'), ('adaptor', 'MASS'), ('pendrive', 'MASS'), ('destroyer', 'SPEED'), ('raccoon', 'SPEED'), ('tummy', 'LENGTH'), ('dragging', 'SPEED'), ('kangaroo', 'SPEED'), ('restaurant', 'LENGTH'), ('forklift', 'SPEED'), ('waddling', 'SPEED'), ('ferry', 'SPEED'), ('mustard', 'MASS'), ('flight', 'SPEED'), ('spider', 'SPEED'), ('parlor', 'LENGTH'), ('peacock', 'SPEED'), ('lime', 'MASS'), ('pigeon', 'SPEED'), ('rolling', 'SPEED'), ('koala', 'MASS'), ('projector', 'MASS'), ('cucumber', 'MASS'), ('adventuring', 'SPEED'), ('cherokee', 'SPEED'), ('jicama', 'MASS'), ('motorbike', 'SPEED'), ('yacht', 'SPEED'), ('firing', 'SPEED'), ('keyboard', 'MASS'), ('scorpion', 'MASS'), ('controller', 'MASS'), ('room', 'LENGTH'), ('scooter', 'SPEED'), ('penthouse', 'LENGTH'), ('mantelpiece', 'LENGTH'), ('finger', 'LENGTH'), ('coriander', 'MASS'), ('raft', 'SPEED'), ('dragonfly', 'SPEED'), ('throwing', 'SPEED'), ('heading', 'SPEED'), ('fish', 'SPEED'), ('hobbling', 'SPEED'), ('butterfly', 'SPEED'), ('cafeteria', 'LENGTH')}
hard_list_objs, _ = zip(*hard_list)
hard_list_objs = set(hard_list_objs)


# %%
BATCH_SIZE = 64


batch = []
batch_num = 0

for j, f in enumerate(files):
    print(f"parsing file: {f}...")
    doq = pd.read_csv(f, names=names, delimiter="\t")
    
    for i, x in tqdm(doq.iterrows()):
        # if x.obj not in hard_list_objs:
        #     continue

        df, flags = parse_raw_counts(x.raw, x.dim)
        
        N = df.counts.sum()

        if x.obj not in hard_list_objs and ("min_bins" in flags
                                            or "min_counts" in flags
                                            or "bin_counts_ratio" in flags):
            ignored.write(f'"{x.obj}"\t"{x.dim}"\t{N}\t{df.shape[0]}\t{j}\n')
            continue

        if len(flags) > 0:
            m = f'"{x.obj}"\t"{x.dim}"\t{"/".join(flags)}\t{df.shape[0]}\t{N}\n'
            flagged.write(m)

        # lower = df['loglower'].values
        # lower[np.abs(lower) > 0.01] = np.round(lower[np.abs(lower) > 0.01], 2)
        # lower[np.abs(lower) > 10.0] = np.round(lower[np.abs(lower) > 10.0], 1)
        # upper = df['logupper'].values
        # upper[np.abs(upper) > 0.01] = np.round(upper[np.abs(upper) > 0.01], 2)
        # upper[np.abs(upper) > 10.0] = np.round(upper[np.abs(upper) > 10.0], 1)
        # to_export = [x.obj,
        #              x.dim, 
        #              lower.tolist(),
        #              upper.tolist(),
        #              df['counts'].values.tolist()]
        
        # batch.append(to_export)

        # if len(batch) == BATCH_SIZE or i == doq.shape[0] - 1:
        #     with open(f"data/batches/batch_{batch_num:05d}.json", "w") as io:
        #         json.dump(batch, io)
        #         batch = []
        #         batch_num += 1

# ignored.close()
flagged.close()


# %%
