from enum import Enum


class Mode(Enum):
    PREDICT = 'predict'
    TEST = 'test'
    TRAIN = 'train'
    PRN = 'prn'


class TrainLayers(Enum):
    META = r"(meta\_.*)"
    RPN = r"(rpn\_.*)"
    RPN_FPN=r"(rpn\_.*)|(fpn\_.*)"
    BRANCH = r"(mrcnn\_.*)"
    HEADS = r"(mrcnn\_.*)|(rpn\_.*)|(meta\_.*)"
    FPN_P = r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(meta\_.*)"
    EXC_BRANCH = r'(?!mrcnn\_).*'
    EXC_RPN = r'(?!rpn\_).*'
    ALL = ".*"


class Model(Enum):
    MRCNN = 'mrcnn'
    FRCNN = 'frcnn'
    META_MRCNN = 'meta_mrcnn'
    META_FRCNN = 'meta_frcnn'


class EvalType(Enum):
    BBOX='bbox'
    SEGM='segm'