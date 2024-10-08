#include <iostream>
using namespace std;

// OOP 3대 요소
// - 상속성 << inheritance (물려받는다..)
// - 은닉성 << data hiding (Encapsulization)
// - 다형성

// 자동차
// 핸들, 페달, 엔진, 전기선, 휘발유
// 구매자 입장에서 사용 : 핸들, 페달, 문
// 몰라도 됨 : 엔진, 전기선

// ====== 접근 지정자 =======
// public, 
// protected(보호받는, 내 자손만 사용 가능), 
// private

// ====== 상속 접근 지정자 ======
// 다음세대한테 부모님의 유산을 어떻게 물려줄지?
// public
// protected 내 자식들까지만 물려줄거얌
// private
class Car {
public:
	void MoveHandle(){}
	void PushPedal(){}
	void OpenDoor(){}	
	void SetHorsePower(int power) {
		m_horsePower = power;
	}
	int GetHorsePower() {
		return m_horsePower;
	}

private:
	// 아래 기능은 사용자가 알아서 안된다. (위험한 걸 공개하는 건 문제가 된다.)
	void Disassemble(){}
	void PutFuelInEngine(){}

private: // 멤버 변수에 직접 접근해서 제어하는 건 정말 별로다. SetHp, GetHp 방법을 사용하는 게 좋다.
	int m_horsePower = 500;
	// 핸들 
	// 페달 
	// 엔진 
	// 문 
	// 전기선
};

int main() {
	Car c;
	
}