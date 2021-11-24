
from libs.datasets import voc as voc


def get_train_dataset(CONFIG, p_split=None):
    if CONFIG.DATASET == 'VOC2012' or CONFIG.DATASET=='VOC2012':
            train_dataset = voc.VOC(
                root=CONFIG.ROOT,
                split='trainaug' if p_split is None else p_split ,
                image_size=CONFIG.IMAGE.SIZE.TRAIN,
                crop_size=CONFIG.IMAGE.SIZE.TRAIN,
                scale=True,
                flip=True,
            )
    else:
        raise ValueError('Dataset name '+str(CONFIG.DATASET) + 'does not match with implemented datasets.')
        return None
    return train_dataset


def get_val_dataset(CONFIG):
    if CONFIG.DATASET == 'VOC2012' or CONFIG.DATASET == 'VOC2012' or CONFIG.DATASET == 'VOC2012_mmi':
            val_dataset = voc.VOC(
                root=CONFIG.ROOT,
                split='val',
                image_size=CONFIG.IMAGE.SIZE.VAL,
                crop_size=CONFIG.IMAGE.SIZE.VAL,
                scale=False,
                flip=False,
            )
    else:
        raise ValueError('Dataset name '+str(CONFIG.DATASET) + 'does not match with implemented datasets.')
        return None
    return val_dataset

