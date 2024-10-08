#include <iostream>

using namespace std;

// 캐스팅 4총사
// static_cast 
// dynamic_cast
// const_cast
// reinterpret_cast

class Player
{
public:
    Player() {};
    virtual ~Player() {};
};
class Knight : public Player{
public:
    Knight(){}
    ~Knight() {}
};
class Dog{};
int main()
{
    // static_cast : 타입 원칙에 비춰볼 때 상식적인 캐스팅만 허용해준다.
    // 1) int <-> float

    int hp = 100;
    int maxHp = 200;

    float ratio = hp / maxHp; // -> 0 why? = 우측의 int 형 계산부터 일어남.
    float ratio1 = (float)hp / maxHp;  // 동일
    float ratio2 = static_cast<float>(hp) / maxHp; // 동일

    Knight* k = new Knight();
    Player* p = k;

    // Knight* k2 = p; // 플레이어는 나이트이다? 처리가 안됨.
    Knight* k3 = (Knight*)p; // 플레이어는 나이트이다? 처리가 안됨.
    // 내가 뭘 하는지 알고 있으니 그냥 캐스팅해줘~
    Knight* k4 = static_cast<Knight*>(p);


    // dynamic_cast : 상속 관계에서 사용하는 캐스팅
    // 다형성을 활용하는 방식 (vitrual)
    // RTTI (Runtime Type Information)
    Knight* k5 = dynamic_cast<Knight*>(p);
    if (k5 != nullptr) {
        // 만약에 Archer :public Player 타입을 Knight로 변환할 때
        // Knight가 아니라면 nullptr을 반환
    }
    // 성능 문제로 지양하는 경우도 있음.

    
    // const_cast -> const 타입을 뗐다 붙였다 할 수 있음.

    // reinterpret_cast : 위험하고, 강력한 형태
    // 다시 생각하다. 
    // 어떤 포인터를 전혀 관계없는 다른 타입으로 변환
    Dog* dog = reinterpret_cast<Dog*>(k5);
    // 너가 들고 있는 메모리를 정수 형태로 가지고 싶어.
    // 그 주소에서 멀지 않은 곳에서 추가적인 연산을 하고 싶어.
}