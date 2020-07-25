import scipy.io as sio
from util import unzipFile
import os
import random


# ============== ShanghaiTech_Crowd_Counting_Dataset V1.0 简介 ================
# Zhang等(Single-image crowd counting via multi-column convolutional neural network)
# 引入了一个新的大规模人群统计数据集，包括1198个图像，330,165个注释头。根据注释人数，数据
# 集是最大的数据集，它包含两部分：A部分和B部分.A部分由482个图像组成，这些图像是从Internet中
# 随机选择的，而B部分是从街道上上海的大都市区。与B部分相比，A部分具有相当大的密度图像。这两
# 部分进一步分为训练和评估集。A部分的训练和测试分别有300和182个图像，而B部分的图像分别有400
# 和316个图像。数据集成功尝试创建具有不同场景类型和不同密度级别的具有挑战性的数据集。然而，各
# 种密度水平的图像数量不均匀，使得训练和评估偏向于低密度水平。然而，该数据集中存在的复杂性，
# 例如不同尺度和透视失真，为更复杂的CNN网络设计创造了新的机会。
# =============================================================================

# =============================================================================
# 该函数转为处理 ShanghaiTech_Crowd_Counting_Dataset V1.0 而实现
# 此函数为预处理函数
# =============================================================================
def pre_ShanghaiTech_Crowd_Counting_Dataset(data_zip_path, unzip_path=None, part='A', sign='train', shuffle=True):
    '''
    data_zip_path  : ShanghaiTech_Crowd_Counting_Dataset.zip的位置, 是个zip文件
    unzip_path     : 解压目录, 是个目录
    label          : 返回数据训练还是测试数据
    sign           : 选择上海数据集的AB哪个部分, 若不明白AB part何意, 可查看本函数之前的注释
    shuffle        : 是否打乱顺序
    '''
    
    # -------------------------------------------------------------------------
    # 文件解压预处理
    # -------------------------------------------------------------------------
    # 获取加压后数据集目录的名字
    data_name = os.path.split(data_zip_path)[1].split('.')[0]

    # 未指定解压目录，则可能已解压在 data_zip_path 同目录下
    if unzip_path is None:

        data_root = data_zip_path[:-4]

        # 如果未找到，则说明未解压
        if not os.path.exists(data_root):
            unzipFile(data_zip_path, unzip_path)

    # 若指定解压目录
    else:

        data_root = os.path.join(unzip_path, data_name)

        # 若该文件目录存在，说明已解压
        if os.path.exists(data_root):
            pass
        else:
            unzipFile(data_zip_path, unzip_path)

    
    # -------------------------------------------------------------------------
    # 选择出准确目录位置, 是 part A 还是 part B
    # -------------------------------------------------------------------------
    AB_list = os.listdir(data_root)

    for dir_AB in AB_list: # 迭代 AB part内容

        if part in dir_AB: # 是否为选择的 part

            temp = os.path.join(data_root, dir_AB)

            if not os.path.isdir(temp): continue
            break

    data_root = temp

    # -------------------------------------------------------------------------
    # 选择出准确目录位置, 是 train 还是 test
    # -------------------------------------------------------------------------
    train_test_list = os.listdir(data_root)

    for tt in train_test_list: # 迭代 train_test_list 内容

        if sign in tt: # 是否为 train 或者 test

            temp = os.path.join(data_root, tt)

            if not os.path.isdir(temp): continue
            break

    data_root = temp

    # -------------------------------------------------------------------------
    # 检查 ground_true 和 image
    # -------------------------------------------------------------------------
    if not 'ground_truth' in os.listdir(data_root):

        raise ValueError('`{}`路径下没有`{}`目录'.format(data_root, 'ground_truth'))

    if not 'images' in os.listdir(data_root):

        raise ValueError('`{}`路径下没有`{}`目录'.format(data_root, 'images'))

    ground_true_dir = os.path.join(data_root, 'ground_truth')
#    images_dir = os.path.join(data_root, 'images')


    # -------------------------------------------------------------------------
    # 将 ground_true 和 image 读到文档中
    # -------------------------------------------------------------------------
    ground_true_list = os.listdir(ground_true_dir)
    if shuffle:
        # 打乱顺序
        random.shuffle(ground_true_list)

    ground_true_list = [os.path.join(ground_true_dir, path) for path in ground_true_list]
    images_list = [ground_true.replace('mat', 'jpg').replace('GT_','').replace('ground_truth', 'images') for ground_true in ground_true_list]

    with open(part + '_' + sign + '_gt.txt', 'w') as f:

        f.write('\n'.join(ground_true_list))

    with open(part + '_' + sign + '_img.txt', 'w') as f:

        f.write('\n'.join(images_list))



# =============================================================================
# 该函数转为处理 ShanghaiTech_Crowd_Counting_Dataset V1.0的 mat文件而实现
# 此函数为预处理函数
# =============================================================================
def deal_ShanghaiTech_Crowd_Counting_Dataset_mat(path):
    '''
    path : 待处理文件的路径
    '''
    # 直接读入该文件是个字典, 有四个键:
    # dict_keys(['__header__', '__version__', '__globals__', 'image_info'])
    data = sio.loadmat(path)
    data = data['image_info']
    
    # 我也不带懂这个 mat 数据为啥没有禁止套娃
    # 获得人头的坐标
    location = data[0][0][0][0][0]
    # 获得总人数
    num = data[0][0][0][0][1][0][0]

    return location, num