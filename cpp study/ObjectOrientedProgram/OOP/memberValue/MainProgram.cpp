#include <iostream>
using namespace std;

// 객체지향
// 데이터 + 로직

class Inventory {
public:
	Inventory() {
		cout << "Inventory()" << endl;
	}
	Inventory(int a) {
		cout << "Inventory(int)" << endl;
	}
	~Inventory() {
		cout << "~Inventory()" << endl;
	}
public:
	int m_itemNum = 10;
};

class Player {
public:
	Player() {
		cout << "Player()" << endl;
	}
	~Player() {
		cout << "~Player()" << endl;
	}
public:

};
class Knight : Player{
public: // 정석적인 문법은  <class Name>() : @@@(100)
	Knight() : m_hp(0)  // 추가로 int& 타입이나 const int 같은 부분은 이 방식을 사용하는 게 좋음.
	/*
		선처리 영역
		사실 이 부분에서 Player의 생성자가 호출됨. 소멸자는 마지막에 호출됨. 빌딩을 허무는 느낌
	*/
	{
		cout << "Knight()" << endl;
		m_hp = 0; // 이렇게도 멤버 변수 초기화 가능
		// m_inventory2 = Inventory(100); // 이중으로 초기화하는 것 주의하기!, 비효율적인 방식이다.
	}
	// 생성자끼리도 호출 가능 Knight(int hp) : m_hp(hp)를 기본 생성자에서 이용하는 방식!
	~Knight() {
		cout << "~Knight()" << endl;
	}
public:
	int m_hp = 0; // 이렇게도 멤버 변수 초기화 가능
	Inventory* m_inventory; // 주소만 가지고 있음.
	Inventory m_inventory2; // 이런 방식은 Knight 데이터 영역에 Inventory 크기만큼 할당받은 상태로 생성됨.
};

int main() {
	Knight k1;


}