#include "Random.h"
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <cmath>
#include <iostream>
#include <cassert>

class Value;
using ValuePtr = std::shared_ptr<Value>;

struct Hash {
    size_t                      operator()(const ValuePtr value) const;
};

class Value : public std::enable_shared_from_this<Value>
{
private:
    inline static size_t        s_currentID{0};
    float                       m_data{};
    float                       m_grad{};
    std::string                 m_op{};
    size_t                      m_id{};
    std::vector<ValuePtr>       m_prev{};
    std::function<void()>       f_backward{};
private:
    Value(float data, const std::string &op, size_t id)
        : m_data{data}, m_op{op}, m_id{id} {}
public:
    const std::string           &getOp()    const { return m_op; }
    float                       getData()   const { return m_data; }
    float                       getGrad()   const { return m_grad; }

    void                        setGrad(float grad) { m_grad = grad; } 
    // don't want clients to call directly to constructor
    // so exposed some API for constructor
    static ValuePtr
    create(float data, const std::string &op = "")
    {
        return ValuePtr(new Value(data, op, Value::s_currentID++));
    }

    ~Value()
    {
        --Value::s_currentID;
    }

    //  out = lhs + rhs
    //  d(out)/d(lhs) = 1; d(out)/d(rhs) = 1
    //  d(L)/d(lhs) = d(L)/d(out) * d(out)/d(lhs) = d(L)/d(out)
    //  d(L)/d(rhs) = d(L)/d(out) * d(out)/d(rhs) = d(L)/d(out)
    static ValuePtr
    add(const ValuePtr &lhs, const ValuePtr &rhs)
    {
        auto out{ Value::create(lhs->m_data + rhs->m_data, "+") };
        out->m_prev = {lhs, rhs};   // push_back(lhs and rhs)
        out->f_backward = [
            lhs_weak = lhs->weak_from_this(),
            rhs_weak = rhs->weak_from_this(),
            out_weak = out->weak_from_this()
        ](){
            // lock() used to temp acquire ownership
            lhs_weak.lock()->m_grad += out_weak.lock()->m_grad;
            rhs_weak.lock()->m_grad += out_weak.lock()->m_grad;
        };
        return out;
    }

    //  out = lhs * rhs
    //  d(out)/d(lhs) = rhs; d(out)/d(rhs) = lhs
    //  d(L)/d(lhs) = d(L)/d(out) * d(out)/d(lhs) -> rhs
    //  d(L)/d(rhs) = d(L)/d(out) * d(out)/d(rhs) -> lhs 
    static ValuePtr
    multiply(const ValuePtr &lhs, const ValuePtr &rhs)
    {
        auto out{ Value::create(lhs->m_data * rhs->m_data, "*") };
        out->m_prev = {lhs, rhs};
        out->f_backward = [
            lhs_weak = lhs->weak_from_this(),
            rhs_weak = rhs->weak_from_this(),
            out_weak = out->weak_from_this() 
        ](){ 
            lhs_weak.lock()->m_grad += out_weak.lock()->m_grad * rhs_weak.lock()->m_data;
            rhs_weak.lock()->m_grad += out_weak.lock()->m_grad *  lhs_weak.lock()->m_data;
        };
        return out;
    }

    //  out = lhs - rhs
    //  d(out)/d(lhs) = 1; d(out)/d(rhs) = -1
    //  d(L)/d(lhs) = d(L)/d(out) * d(out)/d(lhs) = d(L)/d(out)
    //  d(L)/d(rhs) = d(L)/d(out) * d(out)/d(rhs) = -d(L)/d(out)
    static ValuePtr
    subtract(const ValuePtr &lhs, const ValuePtr &rhs)
    {
        auto out{ Value::create(lhs->m_data - rhs->m_data, "-") };
        out->m_prev = {lhs, rhs};
        out->f_backward = [
            lhs_weak = lhs->weak_from_this(),
            rhs_weak = rhs->weak_from_this(),
            out_weak = out->weak_from_this() 
        ](){
            lhs_weak.lock()->m_grad += out_weak.lock()->m_grad;
            rhs_weak.lock()->m_grad -= out_weak.lock()->m_grad;
        };
        return out;
    }

    //  out = base^exponent
    //  d(out)/d(base) = exponent * base^(exponent-1)
    // dL/d(base) = dL/d(out) * d(out)/d(base)
    static ValuePtr
    pow(const ValuePtr &base, float exp)
    {
        float newValue{ std::pow(base->m_data, exp) };
        auto out{ Value::create(newValue, "^") };
        out->m_prev = { base };
        out->f_backward = [
            base_weak = base->weak_from_this(),
            out_weak = out->weak_from_this(),
            exp
        ]() {
            base_weak.lock()->m_grad += out_weak.lock()->m_grad * exp * std::pow(base_weak.lock()->m_data, exp-1);
        };
        return out;
    }

    //  out = num/denum -> out = num * denum^(-1)
    static ValuePtr
    divide(const ValuePtr &num, const ValuePtr &denum)
    {
        auto reciprocal{ pow(denum, -1) }; 
        return Value::multiply(num, reciprocal);
    }

    // out = max(0, inp)
    // when out > 0 : out = inp, so d(out)/d(inp) = 1
    // when out < 0 : out = 0, so d(out)/d(inp) = 0
    // dL/d(inp) = dL/d(out) * d(out)/d(inp)
    static ValuePtr
    relu(const ValuePtr& inp)
    {
        float val{ std::max(0.0f, inp->m_data) };
        auto out{ Value::create(val, "relu") };
        out->m_prev = { inp };

        out->f_backward = [
            inp_weak = inp->weak_from_this(),
            out_weak = out->weak_from_this()
        ]() {
            inp_weak.lock()->m_grad += out_weak.lock()->m_grad * (out_weak.lock()->m_data > 0);
        };
        return out;
    }
    
    //  out = 1/(1+e^(-inp))
    //  d(out)/d(in) = out * (1 - out)
    //  dL/d(inp) = dL/d(out) * d(out)/d(inp)
    static ValuePtr
    sigmoid(const ValuePtr& inp)
    {
        float val{ 1.0f/(1.0f + std::exp(-inp->m_data)) };

        if(inp->m_data < 0)     // prevents overflow when x is highly negative
            val = std::exp(inp->m_data) / (1.0f + std::exp(inp->m_data)); 

        auto out{ Value::create(val, "sigmoid") };
        out->m_prev = { inp };

        out->f_backward = [
            inp_weak = inp->weak_from_this(),
            out_weak = out->weak_from_this(),
            val
        ]() {
            inp_weak.lock()->m_grad += out_weak.lock()->m_grad * val * (1.0f - val);
        };
        return out;
    }

    void
    buildTopo( ValuePtr v, std::unordered_set<ValuePtr, Hash> &visited,
               std::vector<ValuePtr>& topo )
    {
        if(visited.find(v) == visited.end()) {
            visited.insert(v);
            for(const auto &child : v->m_prev)
                buildTopo(child, visited, topo);
            topo.push_back(v);
        }
    }

    void
    backProp()
    {
        std::vector<ValuePtr>               topo{};
        std::unordered_set<ValuePtr, Hash>  visited{};
        buildTopo(shared_from_this(), visited, topo);  // topological sorted graph

        m_grad = 1.0f;

        for (auto it{ topo.rbegin() }; it != topo.rend(); ++it) {
            if(it->get()->f_backward)
                it->get()->f_backward();
            it->get()->print();
        }
    }

    void
    print()
    {
        std::cout << "[data=" << m_data << ", grad=" << m_grad << "]\n";
    }
};

size_t
Hash::operator()(const ValuePtr value) const
{
    return std::hash<std::string>()(value.get()->getOp()) ^ std::hash<float>()(value.get()->getData());
}

// neuron
enum ActivationType {
    relu,
    sigmoid
};

class Activation
{
private:
    static ValuePtr
    relu(const ValuePtr &val)
    {
        return Value::relu(val);
    }

    static ValuePtr
    sigmoid(const ValuePtr &val)
    {
        return Value::sigmoid(val);
    }

public:
    static inline std::unordered_map<ActivationType, std::function<ValuePtr(ValuePtr&)>>
    mActivationFnc = {
        {ActivationType::relu, relu},
        {ActivationType::sigmoid, sigmoid}
    };
};

float
getRandomFloat()
{
   return static_cast<float>(Random::get(-1, 1)) ;
}

class Neuron
{
private:
    std::vector<ValuePtr>       m_weights{};
    ValuePtr                    m_bias{Value::create(0.0)};   
    const ActivationType        m_actt{};       // activation type

public:
    Neuron(size_t len, const ActivationType &actt)
        : m_actt{actt} {
            for (size_t i{0}; i < len; ++i)
                m_weights.emplace_back(Value::create(getRandomFloat()));
    }

    //  for testing
    // Neuron(size_t len, const ActivationType &actt = ActivationType::sigmoid)
    //    : m_actt{actt} {
    //        for (size_t i{0}; i < len; ++i)
    //            m_weights.emplace_back(Value::create(getRandomFloat()));
    //}

    void
    zeroGrad()
    {
        for(auto& w : m_weights)
            w->setGrad(0);
        m_bias->setGrad(0);
    }


    //  Dot product Neuron's weights with the input
    ValuePtr
    operator()(const std::vector<ValuePtr> &x)
    {
        if(x.size() != m_weights.size())
            throw std::invalid_argument("Vectors must be of the same length");

        ValuePtr sum{ Value::create(0.0f) };

        for (size_t i{0}; i < m_weights.size(); ++i) {
            ValuePtr intermedVal{ Value::multiply(x[i], m_weights[i]) };
            sum = Value::add(sum, intermedVal );
        }

        //  Add bias
        sum = Value::add(sum, m_bias);

        const auto &activatinFunc = Activation::mActivationFnc.at(m_actt);
        return activatinFunc(sum);
    }

    size_t getParamSize() const { return m_weights.size() + 1; }

    std::vector<ValuePtr>
    params() const
    {
        std::vector<ValuePtr> out{};
        out.reserve(getParamSize() );

        out.insert(out.end(), m_weights.begin(), m_weights.end());
        out.push_back(m_bias);

        return out;
    }

    void
    printParams() const
    {
        printf("Number of Parameters: %zu\n", getParamSize());
        for (const auto &param : m_weights)
            printf("%f, %f\n", param->getData(), param->getGrad());

        printf("%f, %f\n", m_bias->getData(), m_bias->getGrad());
        std::cout << '\n';
    }

};


class Layer
{
private:
    std::vector<Neuron> m_neurons{};

public:
    Layer(size_t neuronDim, size_t neuronCount, const ActivationType &actt = ActivationType::relu)
    {
        for (size_t i{0}; i < neuronCount; ++i)
            m_neurons.emplace_back( Neuron{neuronDim, actt} );
    }

    std::vector<ValuePtr>
    operator()(const std::vector<ValuePtr> &x)
    {
        std::vector<ValuePtr> out;
        out.reserve(m_neurons.size());
        std::for_each(m_neurons.begin(), m_neurons.end(),
                      [&](auto neuron) {
                        out.emplace_back(neuron(x));
                      });
        return out;
    }

    void
    zeroNeuron()
    {
        for(auto& n : m_neurons)
            n.zeroGrad();
    }

    std::vector<Value*>
    parameters() const
    {
        std::vector<Value*> params{};
        if (params.empty())
            for (const auto &n : m_neurons )
                for (const auto &p : n.params())
                    params.push_back(p.get());
        return params;
    }

    void
    print()
    {
        const auto params{parameters()};
        printf("Num parameters: %d\n", (int)params.size());
        for (const auto& p : params) {
            std::cout << &p << " ";
            printf("[data:%f,grad=%lf]\n", p->getData(), p->getGrad());
        }
        std::cout << '\n';
    }
};

int main()
{
    Layer l1{4, 2};
    l1.print();
}
