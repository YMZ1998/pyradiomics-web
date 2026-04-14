# -- coding: utf-8 --
import SimpleITK as sitk
import os,shutil
#重新制作Mask标签
def createNewMask(file_path,dst_path):
    file_base_name = os.path.basename(file_path)
    sitkImage = sitk.ReadImage(file_path)          #读取nii.gz文件
    npImage = sitk.GetArrayFromImage(sitkImage)    #simpleITK 转换成numpy
    npImage[npImage > 0 ] =1                       #把大于0的标签都变成1，就是所有病区都要
    outImage = sitk.GetImageFromArray(npImage)     #numpy 转换成simpleITK
    outImage.SetSpacing(sitkImage.GetSpacing())    #设置和原来nii.gz文件一样的像素空间
    outImage.SetOrigin(sitkImage.GetOrigin())      #设置和原来nii.gz文件一样的原点位置
    sitk.WriteImage(outImage,os.path.join(dst_path,file_base_name))#保存文件

if not os.path.exists('./Paddle/MyData'):
    os.makedirs('./Paddle/MyData')

source_data_path = './pancreas/'
dst_data_path = './Paddle/MyData/'
kinds = ['MCN','SCN']
index = 0
for kind in kinds:
    img_paths=os.listdir(source_data_path+kind+'CT/')
    seg_paths=os.listdir(source_data_path+kind+'/')
    index=1
    for img_path,seg_path in zip(img_paths,seg_paths):
        result_path=dst_data_path+kind+'/'+str(index).rjust(2,'0')+'/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        print('-'*100)
        print(source_data_path+kind+'CT/'+img_path)
        shutil.copy(source_data_path+kind+'CT/'+img_path,result_path+img_path)
        a,_,_=seg_path.split('.')
        shutil.copy(source_data_path + kind+'/' + seg_path, result_path +a+'_seg'+'.nii.gz')
        index=index+1