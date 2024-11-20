# This file is used to configure the training parameters for each task

class Config_ACDC:
    data_path = "./dataset/cardiac/"
    save_path = "./checkpoints/ACDC/transfuse/convformer/"
    result_path = "./result/ACDC/transfuse/convformer/"
    tensorboard_path = "./tensorboard/ACDC/transfuse/convformer/"
    visual_result_path = "./Visualization/ACDC/transfuse/convformer/"
    load_path = "."
    save_path_code = "_"

    workers = 1                  # number of data loading workers (default: 8)
    epochs = 400                 # number of total epochs to run (default: 400)
    batch_size = 4               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 4                  # the number of classes
    img_size = 256                # the input size of model
    train_split = "trainofficial"        # the file name of training set
    val_split = "valofficial"
    test_split = "testofficial"           # the file name of testing set
    crop = (256, 256)            # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    #eval_mode = "patient_record"
    eval_mode = "patient"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "convformer"



class Config_ISIC:
    data_path = "./dataset/ISIC/"
    save_path = "./checkpoints/ISIC/transfuse/convformer2/"
    result_path = "./result/ISIC/transfuse/convformer2/"
    tensorboard_path = "./tensorboard/ISIC/transfuse/convformer2/"
    visual_result_path = "./Visualization/ISIC/transfuse/convformer2/"
    load_path = "./checkpoints/ISIC/transfuse/convformer2/TransFuse_02030539_154_0.8855464492052707.pth"
    save_path_code = "_"

    workers = 1                  # number of data loading workers (default: 8)
    epochs = 400                 # number of total epochs to run (default: 400)
    batch_size = 4               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes (background + foreground)
    img_size = 256               # the input size of model
    train_split = "train"  # the file name of training set
    val_split = "test"     # the file name of testing set
    test_split = "test"     # the file name of testing set
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "no"                 # the type of input image
    img_channel = 3              # the channel of input image
    eval_mode = "slice"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "transfuse_convformer"
    

# ==================================================================================================
def get_config(task="Synapse"):
    if task == "ACDC":
        return Config_ACDC()
    elif task == "ISIC":
        return Config_ISIC()
