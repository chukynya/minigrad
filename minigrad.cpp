#include "Random.h"
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iterator>
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

//  Hash specialization declaration
struct Hash {
    size_t operator()(const ValuePtr& value) const {
        return std::hash<const Value*>{}(value.get());
    }
};

class Value
    : public std::enable_shared_from_this<Value>
{
private:
    inline static size_t        s_currentID{0};
    Value(float data, const std::string &op, size_t id)
        : m_data{data}, m_op{op}, m_id{id} {}
public:
    const std::string           &getOp()    const { return m_op; }
    float                       getData()   const { return m_data; }
    float                       getGrad()   const { return m_grad; }
    void                        setGrad(float grad) { m_grad = grad; } 
    void                        setData(float data) { m_data = data; }

    void print() { std::cout << "[data=" << getData() << ", grad=" << getGrad() << "]\n"; }

    // don't want clients to call directly to constructor
    // so exposed some API for constructor
    static ValuePtr
    create(float data, const std::string &op = "")
    {
        return ValuePtr(new Value{data, op, Value::s_currentID++});
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
    //  dL/d(base) = dL/d(out) * d(out)/d(base)
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
            base_weak.lock()->m_grad +=
                out_weak.lock()->m_grad * exp *std::pow(base_weak.lock()->m_data, exp-1);
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

    //  out = max(0, inp)
    //  when out > 0 : out = inp, so d(out)/d(inp) = 1
    //  when out < 0 : out = 0, so d(out)/d(inp) = 0
    //  dL/d(inp) = dL/d(out) * d(out)/d(inp)
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

    //  Using Recursive DFS-Based Approach
    void
    buildTopo(ValuePtr v,
              std::unordered_set<ValuePtr, Hash> &visited,
              std::vector<ValuePtr>& topo)
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
        }
    }


private:
    float                       m_data{};
    float                       m_grad{};
    std::string                 m_op{};
    size_t                      m_id{};
    std::vector<ValuePtr>       m_prev{};
    std::function<void()>       f_backward{};
};


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
    activationFnc = {
        {ActivationType::relu, relu},
        {ActivationType::sigmoid, sigmoid}
    };
};

float getRandomFloat() { return static_cast<float>(Random::get(-1, 1)) ; }

class Neuron
{
public:
    Neuron(size_t len, ActivationType actt)
        : m_actt{actt}
    {
        for (size_t i{0}; i < len; ++i)
            m_weights.emplace_back(Value::create(getRandomFloat()));
    }

    //  for testing
    // Neuron(size_t len, ActivationType actt = ActivationType::sigmoid )
    //    : m_actt{actt} {
    //        for (size_t i{0}; i < len; ++i)
    //            m_weights.emplace_back(Value::create(getRandomFloat()));
    //}

    //  + 1 for the bias
    size_t getParamSize() const { return m_weights.size() + 1; }

    //  zero out gradient
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
        //  neuron weight length always same as input length
        if(x.size() != m_weights.size())
            throw std::invalid_argument("Vectors must be of the same length");

        //  one neuron always throw out only one output
        ValuePtr sum{ Value::create(0.0f) };

        for (size_t i{0}; i < m_weights.size(); ++i) {
            ValuePtr intermedVal{ Value::multiply(x[i], m_weights[i]) };
            sum = Value::add(sum, intermedVal );
        }

        //  Add bias
        sum = Value::add(sum, m_bias);

        //  activation function
        const auto &activatinFunc { Activation::activationFnc.at(m_actt) };
        return activatinFunc(sum);
    }


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

private:
    std::vector<ValuePtr>       m_weights{};
    ValuePtr                    m_bias{Value::create(0.0)};   
    const ActivationType        m_actt{};       // activation type
};


class Layer
{
public:
    Layer(size_t neuronDim, size_t neuronCount, const ActivationType &actt = ActivationType::relu)
    {
        for (size_t i{0}; i < neuronCount; ++i)
            m_neurons.emplace_back( Neuron{neuronDim, actt} );
    }

    std::vector<ValuePtr>
    operator()(const std::vector<ValuePtr> &x)
    {
        std::vector<ValuePtr> out{};
        // 1 layer have n neuron also have n output, where n >= 0
        out.reserve(m_neurons.size());
        std::for_each(m_neurons.begin(), m_neurons.end(),
                      [&](auto& neuron) {
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

    std::vector<ValuePtr>
    parameters() const
    {
        std::vector<ValuePtr> params{};
        if (params.empty())
            for (const auto &n : m_neurons )
                for (const auto &p : n.params())
                    params.push_back(p);
        return params;
    }

    void
    print()
    {
        const auto params{ parameters() };
        printf("Num parameters: %d\n", (int)params.size());
        for (const auto& p : params) {
            std::cout << p.get() << " ";
            printf("[data:%f,grad=%lf]\n", p.get()->getData(), p.get()->getGrad());
        }
        std::cout << '\n';
    }

private:
    std::vector<Neuron>     m_neurons{};
};

class Tensor
{
public:
    Tensor(const std::initializer_list<float> &input)
    {
        for (auto val : input) {
            std::vector<ValuePtr> subTensor{};
            subTensor.emplace_back(Value::create(val));
            m_tensor.push_back(subTensor);
        }
    }

    Tensor(const std::initializer_list<std::initializer_list<float>>& input)
    {
        for (const auto& row : input)
        {
            std::vector<ValuePtr> subTensor{};
            subTensor.reserve(row.size());

            for (float val : row)
                subTensor.emplace_back(Value::create(val));

            m_tensor.emplace_back(std::move(subTensor));
        }
    }

    auto begin() { return m_tensor.begin(); }
    auto begin() const { return m_tensor.begin(); }
    auto end() { return m_tensor.end(); }
    auto end() const { return m_tensor.end(); }
    void reset() { m_tensor.clear(); }
    size_t size() const { return m_tensor.size(); }

    void
    zeroNeuron()
    {
        for (auto &subTensor : m_tensor)
            for (auto &val : subTensor)
                val->setGrad(0.0f);
    }

    //  i: row idx
    const std::vector<ValuePtr>&
    operator[](const size_t i) const
    {
        if(m_tensor.size() <= i)
            throw std::invalid_argument("Accessing a Tensor out of bound!");
        return m_tensor[i];
    }

    //  i: row idx
    //  j: col idx
    ValuePtr
    operator()(const size_t i, const size_t j) const
    {
        if(m_tensor.size() <= i)
            throw std::invalid_argument("Accessing a Tensor out of bound!");
        return m_tensor[i][j];
    }

    void
    push_back(const std::vector<ValuePtr> &val)
    {
        std::vector<ValuePtr> subTensor{};
        std::copy(val.begin(), val.end(), std::back_inserter(subTensor));
        m_tensor.emplace_back(subTensor);
    }

private:
    std::vector<std::vector<ValuePtr>>      m_tensor{};
};

class MLP
{
public:
    MLP(size_t inpDim, std::vector<size_t> nouts, const float lr=0.0025)
        :m_lr{lr}
    {
        //  testing: for now assume it's 4 layer
        m_sizes.reserve(4);
        m_sizes.push_back(inpDim);
        //  std::back_inserter safely shifts this behavior from overwriting to inserting:
        //  If you pass a plain m_sizes.begin() iterator to std::copy on an empty container,
        //  it tries to overwrite memory that hasn't been allocated yet, triggering a segfault or undefined behavior.
        std::copy(nouts.begin(), nouts.end(), std::back_inserter(m_sizes));
        for(size_t i{0}; i < m_sizes.size() - 1; ++i)
            m_layers.emplace_back(m_sizes[i],m_sizes[i+1], ActivationType::sigmoid);
    }

    ~MLP() = default;

    void
    zeroGrad()
    {
        for (auto &layer : m_layers)
            layer.zeroNeuron();
    }

    std::vector<ValuePtr>
    parameters() const
    {
        std::vector<ValuePtr> params{};
        if (params.empty())
            for (const auto &n : m_layers )
                for (const auto &p : n.parameters())
                    params.push_back(p);
        return params;
    }

    //  Backprop
    //  W=W-lr * dL/dW
    void
    update()
    {
        for (auto &p : parameters()) {
            p->setData( p->getData() +(float)((float)-m_lr * (float)p->getGrad()));
        }
    }

    //  Forward prop in recursive manner
    std::vector<ValuePtr>
    operator()(const std::vector<ValuePtr> &inp)
    {
        std::vector<ValuePtr> x{inp};
        //  for all the layer, it will get output of the layeri that output
        //  of the layer will become input in the next forward iteration
        for(auto &layer : m_layers) {
            auto y{layer(x)};
            x = y;
        }
        return x;
    }

    void
    printParams() const
    {
        const auto params{parameters()};
        printf("Num parameter: %d\n", (int)params.size());
        for(const auto &p : params) {
            std::cout << &p << " "; 
            printf("[data=%f,grad=%f]\n", p->getData(), p->getGrad());
        }
        printf("\n");
    }

    void save(const std::string& filename) const
    {
        std::ofstream out(filename, std::ios::binary);
        if (!out)
            throw std::runtime_error("Cannot open file for writing.");

        const auto params = parameters();
        size_t count = params.size();

        out.write(reinterpret_cast<const char*>(&count), sizeof(count));

        for (const auto& p : params)
        {
            float value = p->getData();
            out.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
    }

    void load(const std::string& filename)
    {
        std::ifstream in(filename, std::ios::binary);
        if (!in)
            throw std::runtime_error("Cannot open file for loading.");

        size_t count{};
        in.read(reinterpret_cast<char*>(&count), sizeof(count));

        auto params = parameters();

        if (count != params.size())
            throw std::runtime_error("Model architecture mismatch.");

        for (auto& p : params)
        {
            float value{};
            in.read(reinterpret_cast<char*>(&value), sizeof(value));
            p->setData(value);
        }
    }
private:
    std::vector<size_t>     m_sizes{};
    std::vector<Layer>      m_layers{};
    float                   m_lr{};

};

int main()
{
    // 3-input XOR dataset
    Tensor xs{
        {0,0,0},
        {0,0,1},
        {0,1,0},
        {0,1,1},
        {1,0,0},
        {1,0,1},
        {1,1,0},
        {1,1,1}
    };

    std::vector<float> ys{
        0,1,1,0,1,0,0,1
    };

    MLP model(3, {4,4,1}, 0.05f);

    constexpr size_t maxEpochs = 100000;
    constexpr float targetLoss = 0.01f;

    for (size_t epoch = 0; epoch < maxEpochs; ++epoch)
    {
        model.zeroGrad();

        ValuePtr loss = Value::create(0.0f);

        // Forward pass over all samples
        for (size_t i = 0; i < xs.size(); ++i)
        {
            auto pred = model(xs[i]);            // one output
            auto target = Value::create(ys[i]);

            auto diff = Value::subtract(pred[0], target);
            auto sq = Value::pow(diff, 2.0f);

            loss = Value::add(loss, sq);
        }

        // Mean Squared Error
        loss = Value::divide(loss, Value::create(static_cast<float>(xs.size())));

        loss->backProp();
        model.update();

        if (epoch % 1000 == 0)
        {
            std::cout
                << "Epoch " << epoch
                << " | Loss = " << loss->getData()
                << '\n';
        }

        if (loss->getData() < targetLoss)
        {
            std::cout
                << "\nTraining finished at epoch "
                << epoch
                << " (loss = "
                << loss->getData()
                << ")\n";
            break;
        }
    }

    std::cout << "\n=== Final Predictions ===\n";
    std::cout << "Input       Target    Predicted\n";
    std::cout << "--------------------------------\n";

    for (size_t i = 0; i < xs.size(); ++i)
    {
        auto pred = model(xs[i]);

        std::cout
            << xs(i,0)->getData() << ' '
            << xs(i,1)->getData() << ' '
            << xs(i,2)->getData()
            << "        "
            << ys[i]
            << "      "
            << std::fixed << std::setprecision(4)
            << pred[0]->getData()
            << '\n';
    }
    model.save("xor_model.bin");
    std::cout << "\nModel saved as xor_model.bin\n";

    MLP loaded(3, {4,4,1}, 0.00025f);
    loaded.load("xor_model.bin");

    std::cout << "\n=== Loaded Model Predictions ===\n";

    for (size_t i = 0; i < xs.size(); ++i)
    {
        auto pred = loaded(xs[i]);

        std::cout
        << std::fixed << std::setprecision(0)
        << xs(i,0)->getData() << ' '
        << xs(i,1)->getData() << ' '
        << xs(i,2)->getData()
        << "        "
        << ys[i]
        << "      "
        << std::setprecision(4)
        << pred[0]->getData()
        << '\n';
    }

    return 0;

}

