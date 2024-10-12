import diarizationUtils as d

# MODELSETTINGNUMBER == 0이면 model load
# MODELSETTINGNUMBER == 1이면 my train data
# MODELSETTINGNUMBER == 2이면 toy 처음부터 학습
MODEL_SETTING_NUMBER = 3

# model이 저장된 상대위치
MODEL_PATH = "./model.npy"

# 학습할 wav 파일이 저장된 절대위치
# 같은 이름의 json 파일도 필요하다
FILES_FOR_LEARNING_GLOBAL_PATH = "/mnt/my2.2hard/ts5/*.wav"
#FILES_FOR_LEARNING_GLOBAL_PATH = "/mnt/mainStorage/Users/zxc61/Desktop/졸업과제/졸업과제용vscode/diarization/ourRecord.wav"


import sys
import torch
import whisper


def main():

    # 학습용
    train_sequence, train_cluster_id = d.dataPreprocessing(FILES_FOR_LEARNING_GLOBAL_PATH)
    d.modelLoadAndFit(MODEL_PATH)



    # 임시 list 만드는 용도
    """ 
    msqs, cids = d.dataPreprocessing(FILES_FOR_LEARNING_GLOBAL_PATH)
    clusterIdArray = cids.tolist()

    with open ("test_list.txt", "w") as output:
        output.write(str(clusterIdArray))
    """ 

    """ 
    # 임시 list 불러오기
    test_path = './test_list.txt'
    with open(test_path, 'r') as f:
        my = f.read()

    clusterIdArray = eval(my)
    
    if(type(clusterIdArray) is list):
        pass

    else:
        sys.exit("list type error")
    """


    # predict 하는 용도
    
    model, margs, targs, iargs = d.modelLoad(MODEL_PATH)
    msqs = d.dataPreprocessingForPredict(FILES_FOR_LEARNING_GLOBAL_PATH)

    predictedClusterId = 0
    with torch.no_grad():
        predictedClusterId = d.predict(model,msqs,iargs)
        with open ("test_list.txt", "w") as output:
            output.write(str(predictedClusterId))




    clusterIdArray = predictedClusterId

    # Whisper
    model = whisper.load_model("large")



    # hop time :0.01s
    # window 크기 0.02s
    result = model.transcribe("./ourRecord.wav")

    #매핑용
    clusterMappingDict = {}
    clusterMappingDictIndex = 0

    whisperDiarizationOutput = []

    for seg in result['segments']:

        # start 부터 end 까지 시간을 재서
        # cluster id 의 index 자르기
        # 자르기 해서 clustering

        start = seg['start'] * 100

        #hop time = 0.01s
        # start 초 *100 하면 index -> floor
        end = seg['end'] * 100

        _clusterIdDict = {}

        # dictionary 에 나온 횟수 기록
        for i in clusterIdArray[int(start):int(end)]:
            if i in _clusterIdDict:
                _clusterIdDict[i] += 1
            else:
                _clusterIdDict[i] = 1

        # 가장 많이 나온 id 선택
        selectedClusterId = max(_clusterIdDict, key=_clusterIdDict.get)

        if selectedClusterId not in clusterMappingDict:
            clusterMappingDict[selectedClusterId] = clusterMappingDictIndex
            clusterMappingDictIndex += 1
        
        s = f"{seg['start']: .1f}${seg['end']: .1f}$ {clusterMappingDict[selectedClusterId]} ${seg['text']}"
        whisperDiarizationOutput.append(s)
        print(s)

    with open("./whisperDiarizationOutput.txt", 'w') as f:
        f.write('\n'.join(whisperDiarizationOutput))
 



    return None

if __name__ == "__main__":
    main()
