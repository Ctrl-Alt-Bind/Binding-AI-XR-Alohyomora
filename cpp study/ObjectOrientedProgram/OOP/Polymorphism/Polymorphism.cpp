#include <iostream>
using namespace std;


// OOP 3대 요소
// - 상속성 << inheritance (물려받는다..)
// - 은닉성 << data hiding (Encapsulization)
// - 다형성 << Polymorphism

// 오버로딩 vs 오버라이딩
// 오버로딩 = 함수 이름의 재사용
void Test() {}
void Test(float a) {}
void Test(bool b) {}


// 오버라이딩 = 자식 클래스 기능을 재정의하기
// VF table virtual function table
// VF Table [주소1][주소2][주소3] <- virtual 메서드를 사용하면 스택 공간의 첫 번째에 VF Table이 구성된다.
class Player {
public:
	virtual void Move() { cout << "Player Move()!" << endl; } // virtual -> 실행 시점에 결정됐다. Move의 원본 타입을 찾아간다..
	virtual ~Player(){} // Knight 데이터를 정리를 하려면 반드시 부모 클래스의 소멸자에 virtual 키워드를 붙여야 한다.
public:
	int m_hp;
};
// 순수 가상 함수 virtual void Move() = 0; 구현부는 없고 자식 객체에서 작성해야 함.
// 순수 가상 함수를 사용한 경우 추상 클래스라고 칭하며 Move에 대한 구현을 진행해야 한다. (인터페이스 느낌~?)

class Knight : public Player {
public:
	virtual void Move() override { cout << "Knight Move()!" << endl; } // 자식 클래스에 virtual ~~() override 오버라이드 키워드를 반드시 붙이자! 더블체크용도

public:
	int m_stamina;
};

// 바인딩(Binding) = 묶는다.
// 정적 바인딩(static binding) = 컴파일 시점 결정
// 동적 바인딩(dynamic binding) = 실행 시점 결정
void MovePlayer(Player* player) {
	player->Move();
}
int main() {
	Knight k1;
	k1.Move();
	MovePlayer(&k1);
}