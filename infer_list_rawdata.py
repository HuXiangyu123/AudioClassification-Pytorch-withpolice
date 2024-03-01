import os
import argparse
import functools
import pandas as pd
from macls.predict import MAClsPredictor
from macls.utils.utils import add_arguments, print_arguments
#屏蔽报错
import warnings

# 忽略所有警告
warnings.filterwarnings('ignore')

prefix1='CAMPPlus_MFCC'

parser1 = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser1)
add_arg('configs', str, 'configs/cam++_mfcc.yml', '配置文件')
add_arg('use_gpu', bool, True, '是否使用GPU预测')
add_arg('audio_folder', str, 'dataset/784预留0样本', '音频文件夹路径')  # Change to folder path
add_arg('model_path', str, 'models/'+prefix1+'/best_model/', '导出的预测模型文件路径')
args1 = parser1.parse_args()
#print_arguments(args=args)



# 获取识别器
predictor1 = MAClsPredictor(configs=args1.configs,
                           model_path=args1.model_path,
                           use_gpu=args1.use_gpu)

results_df = pd.DataFrame(columns=['Audio_Path', 'Label'])


filefolder_pre='dataset/audios_784_1_784_0/1'
wavfile = []
#读取文件夹下所有wav文件名
for root, dirs, files in os.walk(filefolder_pre):
    for file in files:
        if file.endswith('.wav'):
            wavfile.append(os.path.join(root, file))


#读取audios_228，根据子文件夹是0或1判断实际标签


# 循环预测
# cnt=0
#
# TP = 0
# FP = 0
#
# TN = 0
# FN = 0

for index,audio_path in enumerate(wavfile):
    label_pre1, score = predictor1.predict(audio_data=audio_path)
    #score = (score+score2+score3)/3
    label_pre = label_pre1
    #记录准确率、精度、真假阳性、真假阴性
    # if int(label_pre) == labels[index]:
    #     cnt+=1 #记录正确个数
    #     predict_Ture = "True"
    #     if int(label_pre) == 1:
    #         TP += 1
    #     elif int(label_pre) == 0:
    #         TN += 1
    # else:
    #     predict_Ture = "False"
    #     if int(label_pre) == 1:
    #         FP += 1
    #     elif int(label_pre) == 0:
    #         FN += 1
    results_df = pd.concat([results_df, pd.DataFrame({'Audio_Path': [audio_path], 'Label': [label_pre]})], ignore_index=True)
    if index % 50 == 0:
        print(f'预测进度：{index}/{len(wavfile)}')

# 输出结果）准确率，精度，真假阳性，真假阴性
# print(f"准确率：{cnt/len(labels)}")
# print(f"精度：{TP/(TP+FP)}")
# print(f"真阳性：{TP}")
# print(f"假阳性：{FP}")
# print(f"真阴性：{TN}")
# print(f"假阴性：{FN}")
# 保存结果到Excel
infer_list='正样本测试'
excel_output_path = infer_list+'prediction_results.xlsx'
results_df.to_excel(excel_output_path, index=False)
print(f'Results saved to {excel_output_path}')

