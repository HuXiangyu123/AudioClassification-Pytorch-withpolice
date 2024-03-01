import os
import argparse
import functools
import pandas as pd
from macls.predict import MAClsPredictor
from macls.utils.utils import add_arguments, print_arguments

prefix1='CAMPPlus_MFCC'

parser1 = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser1)
add_arg('configs', str, 'configs/cam++_mfcc.yml', '配置文件')
add_arg('use_gpu', bool, True, '是否使用GPU预测')
add_arg('audio_folder', str, 'dataset/audio_to', '音频文件夹路径')  # Change to folder path
add_arg('model_path', str, 'models/'+prefix1+'/best_model/', '导出的预测模型文件路径')
args1 = parser1.parse_args()
#print_arguments(args=args)

prefix2='CAMPPlus_MelSpectrogram'
parser2= argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser2)
add_arg('configs', str, 'configs/cam++_melspectrogram.yml', '配置文件')
add_arg('use_gpu', bool, True, '是否使用GPU预测')
add_arg('audio_folder', str, 'dataset/audio_to', '音频文件夹路径')  # Change to folder path
add_arg('model_path', str, 'models/'+prefix2+'/best_model/', '导出的预测模型文件路径')
args2 = parser2.parse_args()
#print_arguments(args=args)

prefix3='EcapaTdnn_MFCC'
parser3= argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser3)
add_arg('configs', str, 'configs/ecapa_tdnn_mfcc.yml', '配置文件')
add_arg('use_gpu', bool, True, '是否使用GPU预测')
add_arg('audio_folder', str, 'dataset/audio_to', '音频文件夹路径')  # Change to folder path
add_arg('model_path', str, 'models/'+prefix3+'/best_model/', '导出的预测模型文件路径')
args3 = parser3.parse_args()
#print_arguments(args=args)

# 获取识别器
predictor1 = MAClsPredictor(configs=args1.configs,
                           model_path=args1.model_path,
                           use_gpu=args1.use_gpu)
predictor2 = MAClsPredictor(configs=args2.configs,
                           model_path=args2.model_path,
                           use_gpu=args2.use_gpu)
predictor3 = MAClsPredictor(configs=args3.configs,
                            model_path=args3.model_path,
                            use_gpu=args3.use_gpu)


results_df = pd.DataFrame(columns=['Audio_Path', 'Label', 'Score','predict_Ture'])

#读取labels1.txt (实际标签)
wavfile = []
labels = []
label_file= "labels1.txt"
file = open(label_file, encoding="utf-8-sig", mode="r")
for i in file.read().split("\n"):
    if i:
        filename = "dataset/audio_to/" + i.split("	")[0]
        leibie = int(i.split("	")[1])

        wavfile.append(filename)
        labels.append(leibie)

# 循环预测
cnt=0

TP = 0
FP = 0

TN = 0
FN = 0

for index,audio_path in enumerate(wavfile):
    label_pre1, score = predictor1.predict(audio_data=audio_path)
    label_pre2, score2 = predictor2.predict(audio_data=audio_path)
    label_pre3, score3 = predictor3.predict(audio_data=audio_path)
    score = (score+score2+score3)/3

    labels_pre = [label_pre1, label_pre2, label_pre3]
    label_pre = max(set(labels_pre), key=labels_pre.count)
    #记录准确率、精度、真假阳性、真假阴性
    if int(label_pre) == labels[index]:
        cnt+=1 #记录正确个数
        predict_Ture = "True"
        if int(label_pre) == 1:
            TP += 1
        elif int(label_pre) == 0:
            TN += 1
    else:
        predict_Ture = "False"
        if int(label_pre) == 1:
            FP += 1
        elif int(label_pre) == 0:
            FN += 1
    results_df = pd.concat([results_df, pd.DataFrame({'Audio_Path': [audio_path], 'Label': [label_pre], 'Score': [score],'predict_Ture':[predict_Ture]})], ignore_index=True)
    if index % 50 == 0:
        print(f'预测进度：{index}/{len(wavfile)}')

# 输出结果）准确率，精度，真假阳性，真假阴性
print(f"准确率：{cnt/len(labels)}")
print(f"精度：{TP/(TP+FP)}")
print(f"真阳性：{TP}")
print(f"假阳性：{FP}")
print(f"真阴性：{TN}")
print(f"假阴性：{FN}")
# 保存结果到Excel
excel_output_path = 'stacking'+'prediction_results.xlsx'
results_df.to_excel(excel_output_path, index=False)
print(f'Results saved to {excel_output_path}')

