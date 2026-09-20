#include <algorithm>
#include <unordered_set>
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <functional>
#include <cmath>
#include <iostream>
#include <cassert>
class Value;
using ValuePtr = std::shared_ptr<Value>;

struct Hash
{
    size_t operator()(const ValuePtr value) const;
};

class Value : public std::enable_shared_from_this<Value>
{
//private:
public:
    inline static size_t        s_currentID{0};
    float                       m_data{};
    float                       m_grad{};
    std::string                 m_op{};             // "+";
    size_t                      m_id{};
    std::vector<ValuePtr>       m_prev{};
    std::function<void()>       f_backward{};
private:
    Value(float data, const std::string &op, size_t id)
        : m_data{data}, m_op{op}, m_id{id} {}
public:
    // we don't want clients to call directly to constructor instead to member function
    // a.k.a. we exposed some API for constructor
    static ValuePtr create(float data, const std::string &op = "")
    {
        return ValuePtr(new Value(data, op, Value::s_currentID++));
    }

    ~Value()
    {
        --Value::s_currentID;
    }

    static ValuePtr add(const ValuePtr &lhs, const ValuePtr &rhs)
    {
        auto out{ Value::create(lhs->m_data + rhs->m_data, "+") };
        out->m_prev = {lhs, rhs};
        out->f_backward = [
            lhs_weak = std::weak_ptr<Value>(lhs),
            rhs_weak = std::weak_ptr<Value>(rhs),
            out_weak = std::weak_ptr<Value>(out)
        ](){
            lhs_weak.lock()->m_grad += out_weak.lock()->m_grad;
            rhs_weak.lock()->m_grad += out_weak.lock()->m_grad;
        };
        return out;
    }

    static ValuePtr multiply(const ValuePtr &lhs, const ValuePtr &rhs)
    {
        auto out{ Value::create(lhs->m_data * rhs->m_data, "*") };
        out->m_prev = {lhs, rhs};
        out->f_backward = [
            lhs_weak = std::weak_ptr<Value>(lhs),
            rhs_weak = std::weak_ptr<Value>(rhs),
            out_weak = std::weak_ptr<Value>(out)
        ](){ 
            lhs_weak.lock()->m_grad += rhs_weak.lock()->m_data * out_weak.lock()->m_grad;
            rhs_weak.lock()->m_grad += lhs_weak.lock()->m_data * out_weak.lock()->m_grad;
        };
        return out;
    }

    static ValuePtr subtract(const ValuePtr &lhs, const ValuePtr &rhs)
    {
        auto out{ Value::create(lhs->m_data - rhs->m_data, "-") };
        out->m_prev = {lhs, rhs};
        out->f_backward = [
            lhs_weak = std::weak_ptr<Value>(lhs),
            rhs_weak = std::weak_ptr<Value>(rhs),
            out_weak = std::weak_ptr<Value>(out)
        ](){
            lhs_weak.lock()->m_grad += out_weak.lock()->m_grad;
            rhs_weak.lock()->m_grad -= out_weak.lock()->m_grad;
        };
        return out;
    }

    // out = base^exponent
    // dL/d(base) = dL/d(out) * d(out)/d(base)
    //            = out->grad * exponent * base^(exponent-1)
    static ValuePtr pow(const ValuePtr &base, float exp)
    {
        float newValue{ std::pow(base->m_data, exp) };
        auto out{ Value::create(newValue, "^") };
        out->m_prev = { base };
        out->f_backward = [
            base_weak = std::weak_ptr<Value>(base),
            exp,
            out_weak = std::weak_ptr<Value>(out)
        ]() {
            if(auto base = base_weak.lock())
                base->m_grad += exp * std::pow(base->m_data, exp-1);
        };
        return out;
    }

    static ValuePtr divide(const ValuePtr &num, const ValuePtr &denum)
    {
        auto reciprocal{ pow(denum, -1) }; 
        return Value::multiply(num, reciprocal);
    }

    // f(x) = max(0, x)
    // when x > 0 : y = x, so dy/dx = 1
    // when x < 0 : y = 0, so dy/dx = 0
    static ValuePtr relu(const ValuePtr& inp)
    {
        float val{ std::max(0.0f, inp->m_data) };
        auto out{ Value::create(val, "ReLU") };
        out->m_prev = { inp };
        out->f_backward = [
            inp,
            out
        ]() {
            if(inp)
                // since total gradient is combination of
                // local gradient and outward gradient
                inp->m_grad += (out->m_data > 0) * out->m_grad;
        };
        return out;
    }
    
    // f(x) = 1/(1+e^(-x))
    static ValuePtr sigmoid(const ValuePtr& inp)
    {
        float denum{ 1 + std::exp(-inp->m_data) };
        float val{ 1.0f/denum };
        auto out{ Value::create(val, "sigmoid") };
        out->m_prev = { inp };
        out->f_backward = [
            inp,
            out,
            val
        ]() {
            // f'(x) = f(x) * (1 - f(x))
            inp->m_grad += val * (1.0f - val);
        };
        return out;
    }

    void buildTopo(ValuePtr v,
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

    void backProp()
    {
        std::vector<ValuePtr>               topo{};
        std::unordered_set<ValuePtr, Hash>  visited{};
        buildTopo(shared_from_this(), visited, topo);                    // topological sorted graph

        m_grad = 1.0f;

        for (auto it{ topo.rbegin() }; it != topo.rend(); ++it) {
            if(it->get()->f_backward) {
                it->get()->f_backward();
            }
            it->get()->print();
        }


    }

    void print()
    {
        std::cout << "[data=" << m_data << ", grad=" << m_grad << "]\n";
    }
};

size_t Hash::operator()(const ValuePtr value) const
{
    return std::hash<std::string>()(value.get()->m_op) ^ std::hash<float>()(value.get()->m_data);
}

int main()
{
    auto a{ Value::create(1.0, "") };
    auto b{ Value::create(2.0, "") };

    auto c { Value::add(a, b) };
    std::cout << "c grad: " << c->m_grad << '\n';
    auto d { Value::multiply(c, c) };
    std::cout << "d grad: " << d->m_grad << '\n';

    assert(c->m_data == 3.0);
    assert(c->m_op == "+");

    assert(d->m_data == 9.0);
    assert(d->m_op == "*");

    auto l{ Value::add(d, d) };
    std::cout << "l grad: " << l->m_grad << '\n';
    l->backProp();

    // auto a { std::shared_ptr<Value>() };
}
