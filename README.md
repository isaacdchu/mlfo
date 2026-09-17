# Machine Learning Framework for Othello
Attempt 2 at making a machine learning framework

Todo:
- model constructor by passing in arguments like:
```cpp
Model model = (
    {
        {LinearLayer::factory, {{2}}, {{4}}},
        {ReLULayer::factory, {{4}}, {{4}}}
    },
    MSELoss::factory
);
```
- simplify layers to single input and output path
- convolution operation
- optimizer scheduler?
- namespaces?

