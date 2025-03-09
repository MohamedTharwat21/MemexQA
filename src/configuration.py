

class config:
    def __init__(self):
        self.albumjson=r'/kaggle/input/multimodal-question-answering-1234/album_info.json'   #"path to album_info.json"
        self.qas=r'/kaggle/input/multimodal-question-answering-1234/qas.json' #"path to the qas.json"
        self.glove=r'/kaggle/input/multimodal-question-answering-1234/glove.6B'  #
        #"/path/to img feat npz file"
        self.imgfeat=r'/kaggle/input/multimodal-question-answering-1234/photos_inception_resnet_v2_l2norm.npz'
        self.outpath=r'prepro'   #"output path"
        self.testids=r'/kaggle/input/multimodal-question-answering-1234/test_question.ids'    # "path to test id list"
        self.use_BERT=True
        self.valids = None    #"path to validation id list, if not set will be random 20%% of the training set"
        self.word_based = False   #"Word-based Embedding"
        self.workers = 0
        self.batchSize = 8
        self.niter = 10
        self.lr = 0.01
        self.weight_decay = 5e-6
        self.manualSeed = 1126
        self.mode = 'one-shot'
        self.inpf = './new_dataset/'
        self.outf = './output/'
        self.cuda = True
        self.gpu_id = 0
        self.keep = False
        self.FVTA = False
