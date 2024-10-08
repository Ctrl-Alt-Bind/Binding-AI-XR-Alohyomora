
#include <iostream>

using namespace std;

class Pet {
public:
    Pet() { cout << "Pet()" << endl; }
    ~Pet() { cout << "~Pet()" << endl; }
    Pet(const Pet& pet) { cout << "Pet(const Pet&)" << endl; }
};

class Knight {
public:
    Knight() {

    }
    Knight(const Knight& k) {
        m_hp = k.m_hp;
    }
    void operator=(const Knight& k) {
        m_hp = k.m_hp;
    }
public:
    int m_hp = 0;
    Pet m_pet;
};

int main()
{
    Knight k1;
    k1.m_hp = 200;

    Knight k2 = k1; // 복사 생성자



}

