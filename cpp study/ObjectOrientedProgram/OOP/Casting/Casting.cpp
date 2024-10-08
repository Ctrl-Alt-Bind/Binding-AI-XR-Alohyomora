#include <iostream>

using namespace std;

class Knight {
public:
    int m_hp;

};

class Dog {
public:
    int age; // 0번 오프셋
    int size; // 4번 오프셋
};

int main()
{
    // 포인터의 변환
    Knight* knight = new Knight();

    // 컴파일러는 프로그래머가 어떤 특정 이유에 따라서 Knight를 Dog라는 타입이라고 때를 쓴다고 생각하고, 통과시킨다.    
    Dog* dog = (Dog*)knight; 
    
    // 이 상황에서 dog의 멤버 변수를 건드리면
    // 하필 Player를 상속하는 Knight가 가상 함수를 가지고 있다면 메모리 구조 자체가 달라지기도 한다.
    // 스택 상에서 차지하고 있는 크기가 다를 수 있는데 이를 건드리면 나중에 크래시를 일으킬 수 있다.
    dog->age = 10;
    (*dog).age = 10; // 위 코드와 의미가 같음.

    Knight* k2 = new Knight();











    // C 스타일 캐스팅
    // 값 타입 변환 (명시적 변환)
    // 의미를 유지하기 위해 원본 객체와 다른 비트열 재구성 
    {
        int a = 12345;
        float b = (float)a;
        int c = 0;
    }
        
    // 참조 타입 변환
    {
        int a = 12345;
        float b = (float&)a; // float 타입 비트열로 판단해보니 다른 값이 나와버린다.
        int c = 0;
    }

    // 안전한 변환 (의미가 100% 일치)
    {
        int a = 123456789;
        __int64 b = (__int64)a;
        int c = 0;
    }
    // 불안전한 변환
    {
        int a = 123456789;
        short b = (short)a;
        int c = 0;
    }
    // 암시적 변환
    {
        int a = 123456;
        float b = a; // 컴파일러가 알아서 판단한다. 
    }
}

