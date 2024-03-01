import os
import argparse
import functools
import pandas as pd
from macls.predict import MAClsPredictor
from macls.utils.utils import add_arguments, print_arguments

prefix='CAMPPlus_MFCC'
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs', str, 'configs/cam++.yml', '配置文件')
add_arg('use_gpu', bool, True, '是否使用GPU预测')
add_arg('audio_folder', str, 'dataset/784预留0样本', '音频文件夹路径')  # Change to folder path
add_arg('model_path', str, 'models/'+prefix+'/best_model/', '导出的预测模型文件路径')
args = parser.parse_args()
print_arguments(args=args)

# 获取识别器
predictor = MAClsPredictor(configs=args.configs,
                           model_path=args.model_path,
                           use_gpu=args.use_gpu)

# list
audio_files = [os.path.join(args.audio_folder, f) for f in os.listdir(args.audio_folder) if f.endswith('.wav')]

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
    label_pre, score = predictor.predict(audio_data=audio_path)
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
excel_output_path = prefix+'_'+'prediction_results.xlsx'
results_df.to_excel(excel_output_path, index=False)
print(f'Results saved to {excel_output_path}')

