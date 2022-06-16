# 2022_ai_project_super_resolution

실행방법

    1. kaggle 에서 데이터를 다운 받는다 
    2. data 폴더 안에 img_align_celeba폴더와 list_eval_partition.csv 파일을 넣는다 
    3. *.print, *.test, *.train 파일을 실행한다.

         
폴더 설명 
common 폴더
    function, layer, optimzer ,preprocess 등 학습과정에서 이용할 함수 및 클래스가 선언되어 었습니다.

coclusion/
    3_colors_network/
        3가지의 독립적 네트워크를 구성하여 각 네트워크 의 loss 값의 변화 ,학습된 BGR네트워크를 통한 HR_images 가 들어있습니다.
    
    hard_train/
        Adam optimizer 를 이용하여 학습한 모델을 통해 추출한 HR 이미지 입니다.

    optimizer_compare/
        각 optimzier 로 학습시켰을때 loss 값 변화 추이와 Adam 과 AdaGrad 를 이용해 학습시킨 모델을 통한 HR 이미지가 들어있습니다.
images/
    3_colors_network_images/
        B, G, R 채널별로 독립적으로 학습시킨 모델을 통해 추출한 HR 이미지가 저장되어 있습니다.

    optimizer_compare_images/
        각 optimizer 를 이용하여 학습시킨 4가지모델을 통해 추출한 HR 이미지가 들어있습니다.
        LR 폴더안에는 전처리 후의 이미지가 들어있습니다.(모델 통과 전)
        ANSWER 폴더안에는 모델을 통화시킨 이미지가 들어있습니다.

    output_images/
        Adam 옵티마이저를 사용 하여 hard 하게 학습시킨 모델을 통해 추출된 HR 이미지가 저장됩니다.

data/
    img_align_celeba/
        train, validation, test 이미지들이 들어있습니다.
    
    list_eval_partition.csv
        0, 1, 2 로 train, validation, test 이미지를 구별합니다.

파일 설명 

sr_net.py

1. 초기화 변수
        W1, W2, W3, W4 
            가중치변수로 신경망의 가중치를 설정합니다.
            입력되지 않을시 디폴트 값으로 초기화 됩니다.

2. 인스턴스 변수
        -params
            딕셔너리 변수로, 신경망의 매개변수를 보관합니다.
            params['W1'] 은 1번째 층의 가중치
            params['W2'] 는 2번째 층의 가중치 입니다.

        -layers
            순서가 있는 딕셔너리 변수로, 신경망의 계층을 보관합니다. 
            layers['Conv1'], layers['Relu1'] 와 같이 각 계층을 순서대로 보관합니다.

        -lastLayer  
            신경망의 마지막 계층입니다. 
            Subpixel 계층입니다.

3. 매서드
        - __init__(self, W1, W2, W3, W3) 
            초기화를 수행합니다.

        -predict(self, x)
            예측(추론) 을 수행합니다. 입력 x 는 (B, 1, 44, 44) 저화질 이미지 입니다.

        -loss(self, x, t)
            손실 함수의 값을 구합니다.
            입력 x 는 (B, 1, 44, 44) 저화질 이미지 
            입력 t 는 (B, 1, 176, 176) 의 answer images 입니다.

        -gradient(self, x, t)
            가중치의 매개변수의 기울기를 오차 역전파법으로 구합니다.
            입력 x 는 (B, 1, 44, 44) 저화질 이미지 
            입력 t 는 (B, 1, 176, 176) 의 answer images 입니다.

        -test(self, images, size)
            완성된 모델에 images 를 통과시켜 추론한 HR_images 를 리턴합니다.
            입력 images 는 (B, 216, 176, 3)
            입력 size 는 입력 배치의 수 입니다.

            LR, HR ,ANSWER 이미지를 리턴합니다.

        -test2(self, images, size, model)
            B, G, R 을 독립적으로 학습한 model 객체를 통과시켜 추론한 HR_images 를 리턴 합니다.
            입력 images 는 (B, 216, 176, 3)
            입력 size 는 입력 배치의 수
            입력 model 은  R, G, B 네트워크를 가지고 있는 딕셔너리 모델 변수
            LR, HR ,ANSWER 이미지를 리턴합니다.


optimizer_compare_train.py

    본격적으로 학습하기 전에 손실함수값을 효과적으로 줄여주는 optimizer 를 찾기위한 학습모델입니다.

    train 이미지에서 batch_size 만큼 랜덤으로 뽑아 1000 번 학습시켰습니다.

    4개의 옵티마이저를 사용하였고, 해당되는 학습 네트워크를 네가지 만들어 주었습니다.
    
    optimizer = Adam, SGD, AdaGrad, Momentum



optimzier_compare_test.py
    optimizer_compare_train 에서 각 옵티마이저 별로 학습된 가중치를 이용하여 생성한 HR_images 를  images/optimizer_compare_images 폴더에 옵티마이저 별로 저장합니다.

    images/optimizer_compare_images/LR  폴더의 사진들은 전처리 후의 LR images 입니다
    images/optimizer_compare_images/ANSWER 폴더의 사진들 은 비교할 target 이미지 입니다.

optimizer_compare_print.py
     optimizer_compare_train 에서 각 옵티마이저 별로 학습과정에서 발생한 손실함수의 변화 추이를 conclusion/optimizer_compare 폴더에 저장합니다.

     또한 Adam 과 AdaGrad 를 이용하여 학습시킨 모델에 test 데이터 이미지에서 만든 LR images 를 통과시켜 만든 HR 이미지도 conclusion/optimizer_compare 폴더에 저장합니다.

     conclusion/optimizer_compare 폴더 내의 LR.png는 전처리 후의 사진입니다.

     conclusion/optimizer_compare 폴더 내의 ANSWER.png는 비교할 target 이미지입니다.
     

sr_train.py

   본격적으로 학습하기 위한 모델 입니다. train images 에서 batch_size 만큼 iters_num 만큼 뽑아 학습시킵니다.
   
   목표 학습량은 50000 이었지만, 43114번을 학습시켰습니다.
   
sr_test.py

    sr_train.py 에서 학습시킨 가중치를 가지고 test data의 LR 이미지를  모델에 통과시켜 HR 이미지를 획득합니다. LR, HR, ANSWER 이미지들은  images/output_images 폴더안에 저장됩니다.

sr_print.py

    sr_train.py 에서 학습시킨 가중치를 가지고 test data 의 LR 이미지를 통과시켜 HR 이미지를 획득합니다. LR, HR, ANSWER 이미지를 이용해 conclusion/hard_train 폴더 내에 저장합니다.
            

3_colors_network_train.py

    B, G, R 값을 독립적으로 학습하는 네트워크 신경망을 3가지 만들어 학습시킵니다. train 이미지에서 batch_size 만큼 iters_num번 랜덤으로 뽑아 학습합니다. batch LR_images 의 B 채널은 B채널을 학습하는 네트워크에 들어갑니다. G, R 도 동일하게 해당되는 네트워크에 들어갑니다.

    iters_num 이 25000 까지 학습시켰습니다.

3_colors_network_test.py
    
    3_colors_network_trian.py 를 이용해 얻은 가중치를 이용하여 test data 에서 뽑은 LR 이미지를 각 채널별로 모델이 입력시켜 HR 이미지로 추출합니다. 
    B 채널은 B 채널만 학습한 모델에 
    G 채널은 G 채널만 학습한 모델에 
    R 채널은 R 채널만 학습한 모델에  입력합니다.

    LR 이미지와 HR 이미지 , ANSWER 이미지는 images/3_colors_network_images 폴더내에 각각 저장됩니다.

3_colors_network_print.py

    3_colors_network_train.py 를 수행하는 도중 loss 값의 변화 추이를 plt 를 이용해 시각화 합니다. 
    학습된 3_colors_network 를 이용해 추론한 HR 이미지, LR 이미지 ANSWER 이미지를  conclusion/3_colors_network 폴더에 저장됩니다.


