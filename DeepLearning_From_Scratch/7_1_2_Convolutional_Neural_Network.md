# Chapter 7. Convolutional Neural Network, CNN



## 🐥 I. Structure

- convolution layer
- pooling layer



- CNN

- Fully Connected Layer 인접 계층의 모든 뉴런과 연결됨
    - Affine Layer



## 🐥 II. Convolutional Network

- padding, stride 등의 CNN 고유 용어 등장
- 입체적 데이터(3D Data)



#### Fully Connected Layer

- 인접 계층 뉴런 모두 연결됨
- 출력 수 임의 지정 가능

- 데이터의 형상이 무시됨 모든 입력 데이터를 같은 차원으로 취급
    - 형상에 담긴 정보 살릴 수 없음




#### Convolution Layer

- 입력 3D, 다른 뉴런에게 출력 3D
- 형상 유지
- 이미지처럼 형상을 가진 데이터를 제대로 이해할 가능성 ↑

feature map = inputoutput data

input data of CNN input feature map

output data of CNN output feature map




### Computation

Computation of Convolution Layer - Computation of Filter

Kernel Filter

- 합성곱 연산은 윈도우window를 일정 간격으로 이동해가며 입력 데이터에 적용함.
	- fused multiply-add, FMA

- 필터의 매개변수가 그동안의 가중치에 해당




#### Padding

입력 데이터 주변을 특정 값(0)으로 채움

출력 크기를 조정할 목적으로 사용

- 연산 거칠 때마다 크기 작아지면 출력의 크기 1이 됨 - 합성곱 연산 적용할 수 없는 상황이 됨
  - 이런 상황을 방지하기 위해 Padding 사용, 출력 크기 조정




#### Stride

필터를 적용하는 위치의 간격




#### Output Size

Output Size = (Input + 2  Padding - Filter Size)  Stride + 1

- 각 차원에 대해 각각 연산함

- 정수인 값이여야 시행 가능함
  - 출력 크기가 정수가 되지 않아 오류가 난다면 stride를 변경하거나 정수로 변경하기 위한 솔루션 이용





### Computation of 3 Dim Data


- 길이 방향(채널 방향)으로 특정 맵 늘임

	- 채널 쪽으로 특정 맵이 여러 개 있다면 입력 데이터와 필터의 합성곱 연산을 채널마다 수행


- 입력 데이터의 채널 수 == 필터의 채널 수

	- 필터 자체의 크기는 원하는 값으로 설정 가능

---

*합성곱 연산의 예*


채널 수 C, 높이 H, 너비 W, 필터 높이 FH, 필터 너비 FW


- 입력 데이터 (C, H, W) + 필터 (C, FH, FW) => 출력 데이터 (1, OH, OW)

- 입력 데이터 (C, H, W) + 필터 (FN, C, FH, FW) => 출력 데이터 (FN, OH, OW)
	- 필터를 FN개 적용하면 출력 맵 FN개 생성
	- FN개 맵 모으면 형상 (FN, OH, OW) 블록 완성됨

- 합성곱 연산에서는 **필터 수도 고려**해야 함
	- 필터의 가중치 데이터는 **4차원** 데이터
	- (출력 채널 수, 입력 채널 수, 높이, 너비) 순으로 사용


*편향을 추가한 합성곱 연산의 예*

- 입력 데이터 (C, H, W) + 필터 (FN, C, FH, FW) => (FN, OH, OW) + 편향 (FN, 1, 1) => 출력 데이터 (FN, OH, OW)
	- 편향은 채널 하나에 값 하나로 구성됨
	- 필터 출력 결과와 필터를 더하면, 편향의 각 값이 필터의 출력 블록의 대응 채널 원소 모두에 더해짐


### Batch

합성곱 신경망에서도 배치 처리를 지원

- 각 계층을 흐르는 데이터의 차원을 하나 늘려 4차원 데이터로 저장함

- 신경망에 4차원 데이터가 하나 흐를 때마다 데이터 N개에 대한 합성곱 연산이 이루어짐
	- N회분의 처리 한번에 수행


데이터가 N개일 때 위 예를 배치 처리:

- 입력 데이터(N, C, H, W) + 필터 (FN, C, FH, FW) => (N, FN, OH, OW) + 편향 (FN, 1, 1) => 출력 데이터 (N, FN, OH, OW)











