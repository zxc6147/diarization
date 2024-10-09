import diarizationUtils as d

# MODELSETTINGNUMBER == 0이면 model load
# MODELSETTINGNUMBER == 1이면 my train data
# MODELSETTINGNUMBER == 2이면 toy 처음부터 학습
MODEL_SETTING_NUMBER = 3

# model이 저장된 상대위치
MODEL_PATH = "./model.npy"

# 학습할 wav 파일이 저장된 절대위치
# 같은 이름의 json 파일도 필요하다
FILES_FOR_LEARNING_GLOBAL_PATH = "/media/zxc6147/새 볼륨1/ts5/*.wav"


import sys
import torch
def main():


    #d.dataPreprocessing(FILES_FOR_LEARNING_GLOBAL_PATH)

    #d.modelLoadAndFit(MODEL_PATH)

    model, margs, targs, iargs = d.modelLoad(MODEL_PATH)
    sqs, cids = d.dataPreprocessing(FILES_FOR_LEARNING_GLOBAL_PATH)

    with torch.no_grad():
        d.predict(model, sqs, cids, margs, targs, iargs)

    

    return None

if __name__ == "__main__":
    main()
