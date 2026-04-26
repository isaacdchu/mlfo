#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void update() = 0;
    virtual void step() = 0;
};

#endif // OPTIMIZER_HPP