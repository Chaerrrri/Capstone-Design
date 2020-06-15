# Project title (Capstone design 2020-1)
Deep learning을 이용한 한글 필기체 인식 알고리즘 개선

* Student list
Chaeri Kim

## Overview
* Needs, problems
OCR을 이용한 한글 인쇄체 텍스트의 인식률에 비해 한글 필기체의 인식률은 아직도 높지 않은 편지만, 전체 글자에서 인식 범위를 좁혀 모델을 학습시키는 경우가 많아지고 알고리즘과 학습 환경이 발전함에 따라 특정 범위의 문자에 대한 인식률이 향상되고 있다. 따라서 앞으로의 한글 필기체 인식에서의 과제는 인식 가능한 글자의 범위를 확장 시켜가는 것이 될 것이다. 
이에 본 연구에서는 특정 범위의 문자에 대해 기존의 인식 알고리즘을 변형하여 인식률을 높여보며, 이후 인식하는 글자 범위의 확장을 위해서 인식 모델에 중요한 요소가 무엇인지를 학습해본다.

* Goals, objectives (evaluation)
- PE92 data set에 대하여 클래스 별로 한글 낱글자 이미지를 인식하는 모델 구축
- 비교 논문에서 사용된 모델보다 좀 더 효율적으로 동작하도록 개량해보며 정확도의 최고치인 90% 중후반대에 근접하게 도달하는 것을 목표
- 사용 빈도가 높은 598개 클래스 데이터를 인식하는 모델의 성능 개선

## Results
* Main code, table, graph, comparison, ...
Classes 폴더
- classes_598.txt: 598개 class에 해당하는 [낱글자이름 - 폴더이름] 파일
- files_598.txt: 598개 class에 해당하는 [폴더이름] 파일

Codes 폴더
- converthgu1.py: hgu1 file을 png file로 변환해줌
- make_class_598.py: classes_598.txt를 읽어 폴더이름만 읽어서 files_598.txt 생성 (classes_598.txt이 있는 폴더에서 run)
- make_folder_598.py: classes_598.txt를 읽어 폴더이름만 읽어서 전체 2350개 클래스의 train/test folder 중, 읽었던 598개 클래스에 해당하는 폴더만 복사하여 새로운 598개의 train/test folder 생성 (classes_598.txt이 있는 폴더에서 run)
- data_augmentation.py: 598개 class의 train set에 대하여 data augmentation 적용하여 클래스 별로 30장씩 증량
(Jupyter Notebook에서 run - 경로 지정 유의)
- Model_Augmented_train#.py: 개선된 모델 - Augmented data를 train하고 test (#: 학습 시 train data 내의 train set 비율)
(GPU 서버에서 run - 경로 지정 유의)

References 폴더
- GoogLenet 기반의 딥 러닝을 이용한 향상된 한글 필기체 인식_7.pdf: 참고/비교 대상이었던 논문
- 현대 국어 사용 빈도 조사 2(2005).pdf: 클래스 축소 (사용빈도) 참고자료
- 음절통계_내림차순.xlsx: 음절통계를 빈도수에 따라 내림차순한 엑셀파일 - classes_598.txt 생성 시 활용

Result_capture 폴더
: 실제 test 결과(인식률)를 확인할 수 있음
- Test_Results.hwp: 여러 test 결과를 포함하고 있음

* Web link
PE92 data 참고 (github) https://github.com/daewonyoon/HangulDB
- PE92_train zip file, PE92_test zip file
- Original Source of Codes/converthgu1.py

## Conclusion
* Summary, contribution, ...
개선된 모델의 test accuracy
: 사용 빈도가 높은 598개 클래스의 한글 낱글자 이미지의 학습 결과, 97.13%의 test accuracy를 보였음
(Augmented 데이터가 추가된 69,318개 train set에 대하여 batch size=128, epoch=100으로 train 시)

전체 2350개의 클래스 중에서 598개의 빈출 클래스 데이터에 대해 높은 정확도를 보이는 모델을 구축함을 통해 자주 쓰이는 글자들을 꽤 정확하게 인식해내는 의미 있는 결과를 도출해볼 수 있었다.