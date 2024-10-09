# 할거 메모
# toy n개로 쪼개서 model.fit iteration으로 학습 하기
# model 불러온 후 다음 sequence로 fit 하면 될 듯?
# 이후 aihub data 쪼개서 사용
# predict와 L1, L2, cos 유사도 실행 시간 비교?
# wav 랑 json
# speaker id '?' 인 부분 skip 으로?


import numpy as np
import uisrnn
import time
import glob
import json
import librosa
import os
import sys
import torch, gc


""" 
# MODELSETTINGNUMBER == 0이면 model load
# MODELSETTINGNUMBER == 1이면 my train data
# MODELSETTINGNUMBER == 2이면 toy 처음부터 학습
MODEL_SETTING_NUMBER = 1 """

# iteration 횟수 지정
ITERATION_NUMBER = 1000

#BATCH_SIZE = 10
BATCH_SIZE = 8

# n_mfcc 값, observation dim 값
#N_MFCC_VALUE = 30
N_MFCC_VALUE = 13

FRAME_RATE = 16000

HOP_LENGTH = 160
#HOP_LENGTH = 1600

N_FFT = 320
#N_FFT = 3200


def modelSetting(modelNum):
    """
    # modelNum == 0이면 toy model 단순히 load
    # modelNum == 1이면 my train data 학습
    # modelNum == 2이면 toy 처음부터 학습

    modelSetting(modelNum)
    return 값: train sequence, cluster id, args
    """
    
    print("model setting start")
    start_time = time.time()

    # my train data 불러오기
    if(modelNum == 1):
        train_data = np.load('./my_train_data.npz', allow_pickle=True)

    # 1이 아닌 한 toy training 불러오가
    elif(modelNum == 0 or modelNum == 2):
        train_data = np.load('./toy_training_data.npz', allow_pickle=True)


    test_data = np.load('./toy_testing_data.npz', allow_pickle=True)
    train_sequence = train_data['train_sequence']
    train_cluster_id = train_data['train_cluster_id']
    
    """ 
    test_sequences = test_data['test_sequences'].tolist()
    test_cluster_ids = test_data['test_cluster_ids'].tolist()
    """

    # list로 전달?
    # args[] = uisrnn.parse_arguments()
    model_args, training_args, inference_args = uisrnn.parse_arguments()

    #이터레이션 숫자 지정
    training_args.train_iteration = ITERATION_NUMBER
    training_args.enforce_cluster_id_uniqueness = False
    training_args.batch_size = BATCH_SIZE
    model_args.observation_dim = N_MFCC_VALUE

    print(model_args)
    print(training_args)
    print(inference_args)

    train_sequence = np.array(train_sequence)
    train_sequence = np.squeeze(train_sequence)

    train_cluster_id = np.array(train_cluster_id)
    train_cluster_id = np.squeeze(train_cluster_id)

    
    print(train_sequence.shape)
    print(train_cluster_id.shape)

    #print(train_sequence)
    #print(train_cluster_id)
    
    print(f"model setting done :  {time.time() - start_time}s")

    return train_sequence, train_cluster_id, model_args, training_args, inference_args

def modelInitialization():
    """
    model의 첫 learning(fit)과 save를 수행한다.
    
    modelInitialization(train_sequence, train_cluster_id, model_args, training_args)
    return 값 model
    """

    print("model Learning and save start")
    start_time = time.time()

    model_args, training_args, inference_args = uisrnn.parse_arguments()

    #이터레이션 숫자 지정
    training_args.train_iteration = ITERATION_NUMBER
    training_args.enforce_cluster_id_uniqueness = False
    training_args.batch_size = BATCH_SIZE
    model_args.observation_dim = N_MFCC_VALUE

    # model setting num은 나중에 지우기 -> 분기로 initialization과 load 분리
    model = uisrnn.UISRNN(model_args)
    #model = modelFitSave(model, train_sequence, train_cluster_id, training_args)

    print(f"model initialization done :  {time.time() - start_time}s")

    return model, model_args, training_args, inference_args


def modelFitSave(model, train_sequence, train_cluster_id, training_args):

    """
    model을 인자로 받아 fit하고 save한다.

    modelFitSave(model, train_sequence, train_cluster_id, training_args)
    return 값 model
    """
    print("model Learning and save start")
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()


    # 1~10 10번
    l = len(train_sequence)
    l = (l-1)/10
    for i in range(1, 11):
        t_train_sequence = train_sequence[int(l * (i-1)): int(l*i)]
        print(type(t_train_sequence))
        #t_train_sequence = np.array(t_train_sequence)
        t_train_cluster_id = train_cluster_id[int(l * (i-1)): int(l*i)]
        #t_train_cluster_id = t_train_cluster_id[int(l * (i-1)): int(l*i)]
        model.fit(t_train_sequence, t_train_cluster_id, training_args)

        gc.collect()
        torch.cuda.empty_cache()

    temp_arr = np.array(model)
    np.save('model', temp_arr)

    print(f"model Learning and Save done :  {time.time() - start_time}s")

    return model



def modelLoadAndFit(model_path):
    model, model_args, training_args, inference_args = modelLoad(model_path)
        # my train data 불러오기
    train_data = np.load('./my_train_data.npz', allow_pickle=True)
    train_sequence = train_data['train_sequence']
    train_cluster_id = train_data['train_cluster_id']


    train_sequence = np.array(train_sequence)
    train_sequence = np.squeeze(train_sequence)

    train_cluster_id = np.array(train_cluster_id)
    train_cluster_id = np.squeeze(train_cluster_id)

    """     print(model_args)
    print(training_args)
    print(inference_args)
    print(train_sequence.shape)
    print(train_cluster_id.shape)

    """    
    modelFitSave(model, train_sequence, train_cluster_id, training_args)



def modelLoad(model_path):
    """
    model path를 인자로 받아 model을 load 후 return한다.

    modelLoad(model_path)
    return 값 model
    """

    print("model load start")
    start_time = time.time()

    # model 초기화
    model = 0

    # temp_arr = np.load('./model.npy', allow_pickle=True)
    # model = temp_arr.tolist()
    # 위 두줄을 아래로 축약

    # model load 후 list로 변경
    model = (np.load(model_path, allow_pickle=True)).tolist()

    # sys isexist 로 판별 할까???????????????
    # 예외처리
    if model == 0:
        sys.exit("error, no model file detected")

    model_args, training_args, inference_args = uisrnn.parse_arguments()

    #이터레이션 숫자 지정
    training_args.train_iteration = ITERATION_NUMBER
    training_args.enforce_cluster_id_uniqueness = False
    training_args.batch_size = BATCH_SIZE
    model_args.observation_dim = N_MFCC_VALUE


    # load 후 학습 추가?

    print(f"model load done :  {time.time() - start_time}s")

    return model, model_args, training_args, inference_args





def dataPreprocessing(globalPath):
    """
    global path를 인자로 받는다.
    wav, json data를 가공하여 my_train_data.npz 저장
    """

    print("data processnig start")
    start_time = time.time()

    # wav 파일 저장 위치 global
    file_list = glob.glob(globalPath)


    # 75개 다 불러옴
    print(len(file_list))
    my_test_sequences = []
    my_test_cluster_ids = []



    #파일 진행 퍼센트 확인 >> tqdm으로 가능
    percent = 0

    #파일 불러오기
    for file in file_list:
        my_test_sequence = []

        #wav 파일 librosa로 불러오기
        data_float, sample_rate = librosa.load(file, sr = None)
        global FRAME_RATE
        FRAME_RATE = sample_rate
        data_float = data_float.astype('float64')
        
        # transpose 하면 64개마다 hop만큼 frame
        # sample rate 16,000


        # n_fft 320이면 20 ms. frame length임
        # hop length 160이면 10 ms
        # mfcc /100 하면 시간이 나옴
    
        mfcc = librosa.feature.mfcc(y=data_float, sr=sample_rate, n_fft = N_FFT, n_mfcc=N_MFCC_VALUE, hop_length=HOP_LENGTH)
        mfcc = np.transpose(mfcc)

        print("1111111111")
        print(data_float.shape)
        print(mfcc.shape)


        #print("type은?")
        #print(type(data_float[0]))

        """ 
        for i in range(0, len(data_float), hopLength):
            frame = data_float[i:i+FRAME_SIZE]
            #padding
            if len(frame) < FRAME_SIZE:
                padding_length = FRAME_SIZE - len(frame)
                # zero padding
                frame = np.pad(frame, (0, padding_length), mode='constant')

            frame = np.expand_dims(frame, axis=1)
            my_test_sequence.append(frame)
        """

        my_test_sequence = mfcc
        my_test_sequence = np.array(my_test_sequence)
        my_test_sequence = np.squeeze(my_test_sequence)

        #wav 전체 시간 = 전체 프레임 / sample rate
        #t = len(my_test_sequences) * frame_size / sample_rate
        #print(t)


        """         #my_test_sequence를 3만개씩 쪼개서 넣기
        _index = 0
        while True:
            if(_index + 30000 >= len(my_test_sequence)):
                my_test_sequences.append(my_test_sequence[_index:len(my_test_sequence)])
                break
            else:
                my_test_sequences.append(my_test_sequence[_index:_index+30000])
                _index += 30000
        """

        #같은 이름 다른 확장자
        json_file = os.path.splitext(file)[0]+'.json'
        #print(json_file)

        with open(json_file, 'r', encoding='UTF-8') as f:
            json_data = json.load(f)

        result = []
        for file in json_file:
            # utterance 항목에서 필요한 값들 추출
            for utterance in json_data['utterance']:
                start = utterance['start']
                end = utterance['end']
                speaker_id = utterance['speaker_id'][-1:]  # speaker_id의 뒤자리 추출
            
                # 추출한 데이터로 새로운 딕셔너리 생성
                result.append({
                    'start': start,
                    'end': end,
                    'speaker_id': speaker_id
                })

        my_test_cluster_id = []
        #print("22222222222")
        #print(my_test_sequence.shape)
        ##row len
        #print(len(my_test_sequence))

        """ 
        for i in range(len(my_test_sequence)):
            my_test_cluster_id.append('1')
        """

        # (n_fft / frame rate) / 2, window의 중간 시간
        middleTime = (N_FFT / FRAME_RATE) / 2
        #hop time hop len / frame rate
        hop = HOP_LENGTH / FRAME_RATE

        for dic in result:
            #시작, 끝 시간 float로 표현
            #_f_start = float(dic['start'])
            _f_end = float(dic['end'])

            try:
                #n_speak = '0_' + dic['speaker_id']
                n_speak = dic['speaker_id']
            except ValueError:
                n_speak = '1'

            # start, end speaker Id dictionary로 list 만들고
            # 전체 시간 T = list의 마지막 end 시간
            # sequence 행 1개당 시간은 end 시간 / len(sequence)
            # window의 가운데에 해당하는 speaker id를 갖자.
            
            # (nf = start)초일때 배열 my_test_cluster_id[N]
            # nt = end
            # array_n_start = floor(len(my_test_cluster_id) * n_start / t)
            # array_n_end = floor(len(my_test_cluster_id) * n_end / t)
            # my_test_cluster_id[Nf]~[Nt] speaker_id(n_speak)로 초기화

            """ 
            t = len(my_test_sequence) * hopLength / sample_rate
            array_n_start = math.floor(len(my_test_cluster_id) * n_start / t)
            array_n_end = math.floor(len(my_test_cluster_id) * n_end / t)

            while array_n_start <= array_n_end:
                my_test_cluster_id[array_n_start] = n_speak
                array_n_start += 1
            """

            # time > end iter 돌리고
            # 만약 time > end 면 다음 dic로 간다.

            while middleTime <= _f_end:
                my_test_cluster_id.append(n_speak)
                middleTime += hop

        # len(my_test_sequence) vs len(my_test_cluster_id)

        # test cluster id 갯수 안맞을 경우 보정
        if len(my_test_cluster_id) < len(my_test_sequence):
            for _ in range(len(my_test_sequence) - len(my_test_cluster_id)):
                my_test_cluster_id.append(my_test_cluster_id[-1])




        # n_fft 320이면 20 ms. frame length임 -> window 크기
        # hop length 160이면 10 ms
        # mfcc /100 하면 시간이 나옴
        # mfcc(y=data_float, sr=sample_rate, n_fft = 320, n_mfcc=N_MFCC_VALUE, hop_length=hopLength) 
        # window만큼의 wav를 특징 벡터로 변환, 그다음 hop만큼 건너뛴다

        # 중간 시간을 second로 변환하자면, fft / framerate -> window length이고
        # hop / framerate = hop time
        # hop * n + 1/2 window len 가 n + 1번째 중간 시간
        # sequence랑 cluster id len 비교
        # sequence가 더 크면 cluster id 의 마지막 index 복제해서 append 해주기

        # window 중간 시간의 speaker num를 speaker number로 가지자.
        # 중간 시간에 hop 만큼 더해가면서 json의 end 시간과 비교
        # json의 end 시간보다 넘으면 다음 dictionary 읽기


        #print(np.asarray(my_test_cluster_id).shape)
        """ 
        #my_test_cluster를 3만개씩 쪼개서
        _index = 0
        while True:
            if(_index + 30000 >= len(my_test_cluster_id)):
                my_test_cluster_ids.append(my_test_cluster_id[_index:len(my_test_cluster_id)])
                break
            else:
                my_test_cluster_ids.append(my_test_cluster_id[_index:_index+30000])
                _index += 30000

        """
        percent+=1
        print("percentage : ", percent / len(file_list) * 100)

        if percent == 1:
            break 

    print("wav파일 모두 읽어서 배열에 저장한 시간 : ", time.time() - start_time)


    my_test_sequences = my_test_sequence
    my_test_cluster_ids = my_test_cluster_id

    my_test_sequences = np.array(my_test_sequences)
    my_test_sequences = np.squeeze(my_test_sequences)

    my_test_cluster_ids = np.array(my_test_cluster_ids)
    my_test_cluster_ids = np.squeeze(my_test_cluster_ids)



    print(my_test_sequences.shape)
    print(my_test_cluster_ids.shape)

    np.savez('my_train_data.npz', train_sequence=my_test_sequences, train_cluster_id=my_test_cluster_ids)

    print("저장한 np array들을 npz로 저장한 시간 : ", time.time() - start_time)

    return my_test_sequences, my_test_cluster_ids





def predict(model, test_sequences, test_cluster_ids, model_args, training_args, inference_args):
    print("predict start")
    start_time = time.time()
    predicted_cluster_ids = []
    test_record = []

    i = 0
    """     for(test_sequence, test_cluster_id) in zip(test_sequences, test_cluster_ids):
        print(test_sequence.shape)

        
        predicted_cluster_id = model.predict(test_sequence, inference_args)
        predicted_cluster_ids.append(predicted_cluster_id)
        accuracy = uisrnn.compute_sequence_match_accuracy(test_cluster_id, predicted_cluster_id)
        test_record.append((accuracy, len(test_cluster_id)))
        print('Ground truth labels: ')
        print(test_cluster_id)
        print('Predicted labels: ')
        print(predicted_cluster_id)
        print('-' * 100)
        i+=1 """
    predicted_cluster_id = model.predict(test_sequences, inference_args)
    print(f"predict done :  {time.time() - start_time}s")
    predicted_cluster_ids.append(predicted_cluster_id)
    print(f"Append done :  {time.time() - start_time}s")
    print(predicted_cluster_id)
    print(type(predicted_cluster_id))
    print(type(test_cluster_ids))
    accuracy = uisrnn.compute_sequence_match_accuracy(test_cluster_ids.tolist(), predicted_cluster_id)
    print(f"Accuracy done :  {time.time() - start_time}s")
    test_record.append((accuracy, len(test_cluster_ids)))
    print('Ground truth labels: ')
    print(test_cluster_ids)
    print('Predicted labels: ')
    print(predicted_cluster_id)

    output_result = uisrnn.output_result(model_args, training_args, test_record)
    print(output_result)  
    print(f"All predict done :  {time.time() - start_time}s")


