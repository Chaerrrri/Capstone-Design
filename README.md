# Project title (Capstone design 2020-1)
Deep learning�� �̿��� �ѱ� �ʱ�ü �ν� �˰��� ����

* Student list
Chaeri Kim

## Overview
* Needs, problems
OCR�� �̿��� �ѱ� �μ�ü �ؽ�Ʈ�� �νķ��� ���� �ѱ� �ʱ�ü�� �νķ��� ������ ���� ���� ������, ��ü ���ڿ��� �ν� ������ ���� ���� �н���Ű�� ��찡 �������� �˰���� �н� ȯ���� �����Կ� ���� Ư�� ������ ���ڿ� ���� �νķ��� ���ǰ� �ִ�. ���� �������� �ѱ� �ʱ�ü �νĿ����� ������ �ν� ������ ������ ������ Ȯ�� ���Ѱ��� ���� �� ���̴�. 
�̿� �� ���������� Ư�� ������ ���ڿ� ���� ������ �ν� �˰����� �����Ͽ� �νķ��� ��������, ���� �ν��ϴ� ���� ������ Ȯ���� ���ؼ� �ν� �𵨿� �߿��� ��Ұ� ���������� �н��غ���.

* Goals, objectives (evaluation)
- PE92 data set�� ���Ͽ� Ŭ���� ���� �ѱ� ������ �̹����� �ν��ϴ� �� ����
- �� ������ ���� �𵨺��� �� �� ȿ�������� �����ϵ��� �����غ��� ��Ȯ���� �ְ�ġ�� 90% ���Ĺݴ뿡 �����ϰ� �����ϴ� ���� ��ǥ
- ��� �󵵰� ���� 598�� Ŭ���� �����͸� �ν��ϴ� ���� ���� ����

## Results
* Main code, table, graph, comparison, ...
Classes ����
- classes_598.txt: 598�� class�� �ش��ϴ� [�������̸� - �����̸�] ����
- files_598.txt: 598�� class�� �ش��ϴ� [�����̸�] ����

Codes ����
- converthgu1.py: hgu1 file�� png file�� ��ȯ����
- make_class_598.py: classes_598.txt�� �о� �����̸��� �о files_598.txt ���� (classes_598.txt�� �ִ� �������� run)
- make_folder_598.py: classes_598.txt�� �о� �����̸��� �о ��ü 2350�� Ŭ������ train/test folder ��, �о��� 598�� Ŭ������ �ش��ϴ� ������ �����Ͽ� ���ο� 598���� train/test folder ���� (classes_598.txt�� �ִ� �������� run)
- data_augmentation.py: 598�� class�� train set�� ���Ͽ� data augmentation �����Ͽ� Ŭ���� ���� 30�徿 ����
(Jupyter Notebook���� run - ��� ���� ����)
- Model_Augmented_train#.py: ������ �� - Augmented data�� train�ϰ� test (#: �н� �� train data ���� train set ����)
(GPU �������� run - ��� ���� ����)

References ����
- GoogLenet ����� �� ������ �̿��� ���� �ѱ� �ʱ�ü �ν�_7.pdf: ����/�� ����̾��� ��
- ���� ���� ��� �� ���� 2(2005).pdf: Ŭ���� ��� (����) �����ڷ�
- �������_��������.xlsx: ������踦 �󵵼��� ���� ���������� �������� - classes_598.txt ���� �� Ȱ��

Result_capture ����
: ���� test ���(�νķ�)�� Ȯ���� �� ����
- Test_Results.hwp: ���� test ����� �����ϰ� ����

* Web link
PE92 data ���� (github) https://github.com/daewonyoon/HangulDB
- PE92_train zip file, PE92_test zip file
- Original Source of Codes/converthgu1.py

## Conclusion
* Summary, contribution, ...
������ ���� test accuracy
: ��� �󵵰� ���� 598�� Ŭ������ �ѱ� ������ �̹����� �н� ���, 97.13%�� test accuracy�� ������
(Augmented �����Ͱ� �߰��� 69,318�� train set�� ���Ͽ� batch size=128, epoch=100���� train ��)

��ü 2350���� Ŭ���� �߿��� 598���� ���� Ŭ���� �����Ϳ� ���� ���� ��Ȯ���� ���̴� ���� �������� ���� ���� ���̴� ���ڵ��� �� ��Ȯ�ϰ� �ν��س��� �ǹ� �ִ� ����� �����غ� �� �־���.