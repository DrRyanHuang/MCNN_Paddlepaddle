import zipfile
import os
import json





#===========================================================================================
# 文件解压函数
#===========================================================================================
def unzipFile(zipfile_path, out_path=None):
    '''
    * zipfile_path   :   未解压文件路径
    * out_path       :   输出文件目录(默认与zip file同路径，同名字)
    '''
    # 压缩文件检验
    assert zipfile.is_zipfile(zipfile_path), '该文件不是有效 zip 压缩文件'
    
    try:
        # 给默认值赋值
        if out_path == None:
            out_path = zipfile_path[:-4]
            
        # 创建 ZipFile 对象
        zip_file = zipfile.ZipFile(zipfile_path)
        
        # 解压缩
        zip_file.extractall(out_path)
        zip_file.close()
        
    except:
        
        print('文件 {} 解压缩失败，请查看 `unzipFile` 函数'.format(zipfile_path))
        return False
    
    return True




#===========================================================================================
# 读取 json 文件
#===========================================================================================
def get_pic_num_from_json(json_path):
    # 从所给json文件中得到预测list
    # json_path : json文件path

    with open(json_path, 'r') as f:
        json_dic = json.load(f)
    
    return json_dic