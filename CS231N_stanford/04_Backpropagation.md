# Chapter 04. Backpropagation

*if the input: f and output: q in backpropagation, the gradient of q means the effect of q on f*


## 🐶 Review

- Loss Function
  - show how bad the model is
  - ex) SVM loss

- Optimization
  - to find W that minimalize loss(result of loss function)
 
```python
# Vanilla Gradient Descent
 
while True:
  weights_grad = evaluate_gradient(losS_fun, data, weights)
  weights += - step_size * weights_grad
```

- Gradient Descent

    - find gradient about W
    - Numerical Gradient: slow
    - Analytic Gradient: fast
      - 미분으로 공식을 유도하여 gradient 계산

Gradient = 다변수 함수의 모든 입력값에서 모든 방향에 대한 순간 변화율 = 편미분값의 벡터




## 🐶 Computational Graph

: useful to catch the flow of front-back propagation

sigmoid gate: grouping

*gradient descent of end node: 1*


### Gate

**add** gate: gradient distributor / local gradient * upstream gradient

**mul** gate: swap multiplier / local gradient * upstream gradient of another input node

**copy** gate: gradient adder / output1 local gradient + output2 local gradient

**max** gate: gradient router / local gradient 받은 쪽으로 upstream gradient 전달, 반대쪽은 0



## 🐶 Backpropagation

: recursive application of the chain rule along a computational graph to compute the gradients of all inputs/parameters/intermediates

- forward: compute & save

- backward: apply the chain rule to compute the gradient of the loss function with respect to the inputs

변수의 gradient는 변수와 같은 shape 가져야 함 -> *check gradient shape == variable shape*


### Jacobian

derivative

diagonal matrix: 출력의 해당 요소에만 영향을 줌


