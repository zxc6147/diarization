import diarizationUtils as d

# MODELSETTINGNUMBER == 0이면 model load
# MODELSETTINGNUMBER == 1이면 my train data
# MODELSETTINGNUMBER == 2이면 toy 처음부터 학습
MODEL_SETTING_NUMBER = 1

# model이 저장된 상대위치
MODEL_PATH = "./model.npy"

# 학습할 wav 파일이 저장된 절대위치
# 같은 이름의 json 파일도 필요하다
FILES_FOR_LEARNING_GLOBAL_PATH = "/media/zxc6147/새 볼륨1/ts5/*.wav"

def main():

    #d.dataPreprocessing(FILES_FOR_LEARNING_GLOBAL_PATH)
    ts, tci, ma, ta, ia = d.modelSetting(1)
    model = d.modelInitialization(ts, tci, ma, ta)

    return None

if __name__ == "__main__":
    main()
